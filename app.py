import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
import requests
from difflib import get_close_matches

st.set_page_config(page_title="Logiciel de Prédiction Pro v4 (Live Odds API)", layout="wide")


def read_api_key_from_secrets():
    """Lit la clé dans les secrets Streamlit (Cloud ou .streamlit/secrets.toml), sinon vide."""
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""


DEFAULT_API_KEY = read_api_key_from_secrets()

# --- 1. CONFIGURATION DES CHAMPIONNATS ---
LEAGUES = {
    "Ligue 1 (France)": {"code": "F1", "odds_api": "soccer_france_ligue_one"},
    "Premier League (Angleterre)": {"code": "E0", "odds_api": "soccer_epl"},
    "La Liga (Espagne)": {"code": "SP1", "odds_api": "soccer_spain_la_liga"},
    "Serie A (Italie)": {"code": "I1", "odds_api": "soccer_italy_serie_a"},
    "Bundesliga (Allemagne)": {"code": "D1", "odds_api": "soccer_germany_bundesliga"}
}

SEASONS = {
    "Multi-saisons (2022 à 2027)": ["2223", "2324", "2425", "2526", "2627"],
    "2 dernières saisons": ["2526", "2627"]
}

# --- 2. COLLECTE DE L'HISTORIQUE ---
@st.cache_data(ttl=3600)
def load_historical_data(league_code, season_codes):
    frames = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for s in season_codes:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                if all(c in df.columns for c in cols):
                    df = df[cols].dropna()
                    df['Season'] = s
                    frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    
    full_df = pd.concat(frames, ignore_index=True)
    full_df['FTHG'] = full_df['FTHG'].astype(int)
    full_df['FTAG'] = full_df['FTAG'].astype(int)
    full_df['Date_dt'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
    full_df = full_df.dropna(subset=['Date_dt']).sort_values('Date_dt').reset_index(drop=True)
    return full_df

# --- 3. HARMONISATION DES NOMS D'ÉQUIPES (API <-> HISTORIQUE) ---
TEAM_ALIASES = {
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "Stade Brestois 29": "Brest",
    "Stade Rennais": "Rennes",
    "RC Lens": "Lens",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "RB Leipzig": "RB Leipzig"
}

def match_team_name(name, valid_teams):
    clean = TEAM_ALIASES.get(name, name)
    if clean in valid_teams:
        return clean
    for vt in valid_teams:
        if vt.lower() in clean.lower() or clean.lower() in vt.lower():
            return vt
    matches = get_close_matches(clean, valid_teams, n=1, cutoff=0.45)
    return matches[0] if matches else None

# --- 4. CONNEXION TEMPS RÉEL (THE ODDS API EN PRIORITÉ 1) ---
@st.cache_data(ttl=300)
def fetch_upcoming_matches(league_code, sport_key, odds_api_key, valid_teams, df_history):
    """
    Priorité 1 : The Odds API en Temps Réel (si une clé est fournie).
    Priorité 2 : Flux officiel Football-Data (fixtures.csv).
    Priorité 3 : Calendrier restant déduit automatiquement.
    """
    upcoming = []
    api_quota_info = None
    headers = {"User-Agent": "Mozilla/5.0"}

    # PRIORITÉ 1 : THE ODDS API (TEMPS RÉEL)
    if odds_api_key and len(odds_api_key.strip()) > 10:
        try:
            url_api = (
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
                f"?apiKey={odds_api_key.strip()}&regions=eu&markets=h2h&oddsFormat=decimal"
            )
            res = requests.get(url_api, timeout=10)
            rem = res.headers.get("x-requests-remaining", "?")
            used = res.headers.get("x-requests-used", "?")
            api_quota_info = f"🟢 The Odds API connectée en direct | Requêtes restantes ce mois : {rem} (utilisées : {used})"

            if res.status_code == 200:
                for item in res.json():
                    raw_h = item.get('home_team', '')
                    raw_a = item.get('away_team', '')
                    h = match_team_name(raw_h, valid_teams)
                    a = match_team_name(raw_a, valid_teams)
                    if h and a and h != a:
                        odds_1, odds_n, odds_2 = [], [], []
                        bookmaker_count = len(item.get('bookmakers', []))
                        for bm in item.get('bookmakers', []):
                            for market in bm.get('markets', []):
                                if market.get('key') == 'h2h':
                                    for out in market.get('outcomes', []):
                                        if out['name'] == raw_h:
                                            odds_1.append(float(out['price']))
                                        elif out['name'] == raw_a:
                                            odds_2.append(float(out['price']))
                                        else:
                                            odds_n.append(float(out['price']))
                        
                        commence = item.get('commence_time', '')
                        upcoming.append({
                            "Date": commence[:10] if commence else "À venir",
                            "Heure": commence[11:16] if len(commence) >= 16 else "",
                            "HomeTeam": h,
                            "AwayTeam": a,
                            "Odd_1": round(max(odds_1), 2) if odds_1 else np.nan,
                            "Odd_N": round(max(odds_n), 2) if odds_n else np.nan,
                            "Odd_2": round(max(odds_2), 2) if odds_2 else np.nan,
                            "Avg_1": round(np.mean(odds_1), 2) if odds_1 else np.nan,
                            "Avg_N": round(np.mean(odds_n), 2) if odds_n else np.nan,
                            "Avg_2": round(np.mean(odds_2), 2) if odds_2 else np.nan,
                            "Bookmakers": bookmaker_count,
                            "Source": "⚡ The Odds API (Temps Réel)"
                        })
            elif res.status_code == 401:
                api_quota_info = "🔴 Clé The Odds API invalide ou expirée (Erreur 401). Bascule sur le flux de secours."
            elif res.status_code == 429:
                api_quota_info = "🟠 Quota mensuel The Odds API atteint (Erreur 429). Bascule sur le flux de secours."
        except Exception as e:
            api_quota_info = f"🟠 Erreur réseau API ({str(e)[:40]}). Bascule sur le flux de secours."

    # PRIORITÉ 2 : FLUX OFFICIEL FOOTBALL-DATA (SI PAS DE CLÉ API OU AUCUN MATCH RETOURNÉ)
    if not upcoming:
        try:
            url_fix = "https://www.football-data.co.uk/fixtures.csv"
            res = requests.get(url_fix, headers=headers, timeout=8)
            if res.status_code == 200:
                df_fix = pd.read_csv(io.StringIO(res.text))
                if 'Div' in df_fix.columns:
                    df_fix = df_fix[df_fix['Div'] == league_code]
                    for _, row in df_fix.iterrows():
                        h = match_team_name(str(row.get('HomeTeam', '')), valid_teams)
                        a = match_team_name(str(row.get('AwayTeam', '')), valid_teams)
                        if h and a and h != a:
                            odd_h = row.get('AvgH', row.get('B365H', np.nan))
                            odd_d = row.get('AvgD', row.get('B365D', np.nan))
                            odd_a = row.get('AvgA', row.get('B365A', np.nan))
                            upcoming.append({
                                "Date": str(row.get('Date', 'À venir')),
                                "Heure": str(row.get('Time', '')),
                                "HomeTeam": h,
                                "AwayTeam": a,
                                "Odd_1": float(odd_h) if pd.notna(odd_h) else np.nan,
                                "Odd_N": float(odd_d) if pd.notna(odd_d) else np.nan,
                                "Odd_2": float(odd_a) if pd.notna(odd_a) else np.nan,
                                "Bookmakers": 1,
                                "Source": "Flux Officiel Football-Data"
                            })
        except Exception:
            pass

    # PRIORITÉ 3 : GÉNÉRATION AUTOMATIQUE DES MATCHS RESTANTS DE LA SAISON
    if not upcoming and not df_history.empty:
        latest_season = df_history['Season'].iloc[-1]
        df_current = df_history[df_history['Season'] == latest_season]
        current_teams = sorted(list(set(df_current['HomeTeam']).union(set(df_current['AwayTeam']))))
        played_pairs = set(zip(df_current['HomeTeam'], df_current['AwayTeam']))
        
        used_teams = set()
        for h in current_teams:
            if h in used_teams:
                continue
            for a in current_teams:
                if h != a and a not in used_teams and (h, a) not in played_pairs:
                    upcoming.append({
                        "Date": "Prochaine Journée",
                        "Heure": "Auto",
                        "HomeTeam": h,
                        "AwayTeam": a,
                        "Odd_1": np.nan,
                        "Odd_N": np.nan,
                        "Odd_2": np.nan,
                        "Bookmakers": 0,
                        "Source": "Calendrier Restant Déduit"
                    })
                    used_teams.add(h)
                    used_teams.add(a)
                    break

    return pd.DataFrame(upcoming), api_quota_info

# --- 5. MOTEUR MATHÉMATIQUE (ELO + FORME 5 MATCHS + POISSON) ---
def calculate_elo_ratings(df, k_factor=20, home_advantage=65):
    teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
    elo = {team: 1500.0 for team in teams}
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        hg, ag = row['FTHG'], row['FTAG']
        r_h, r_a = elo[h], elo[a]
        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + home_advantage)) / 400.0))
        score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        goal_diff = abs(hg - ag)
        margin_mult = np.log(goal_diff + 1) if goal_diff > 0 else 1.0
        delta = k_factor * margin_mult * (score_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
    return elo

def get_recent_form(df, team, n_matches=5):
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(n_matches)
    if team_matches.empty:
        return 1.0, 1.0, ""
    gf_list, ga_list, streak = [], [], []
    for _, row in team_matches.iterrows():
        gf, ga = (row['FTHG'], row['FTAG']) if row['HomeTeam'] == team else (row['FTAG'], row['FTHG'])
        gf_list.append(gf)
        ga_list.append(ga)
        streak.append("🟢V" if gf > ga else ("🟡N" if gf == ga else "🔴D"))
    return np.mean(gf_list), np.mean(ga_list), " ".join(streak)

def calculate_weighted_strengths(df, form_weight=0.35, n_recent=5):
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    avg_total = (avg_home_goals + avg_away_goals) / 2.0

    home_stats = df.groupby('HomeTeam').agg(goals_for_h=('FTHG', 'mean'), goals_against_h=('FTAG', 'mean'))
    away_stats = df.groupby('AwayTeam').agg(goals_for_a=('FTAG', 'mean'), goals_against_a=('FTHG', 'mean'))
    stats = pd.concat([home_stats, away_stats], axis=1).fillna(avg_total)

    stats['att_home_hist'] = stats['goals_for_h'] / avg_home_goals
    stats['def_home_hist'] = stats['goals_against_h'] / avg_away_goals
    stats['att_away_hist'] = stats['goals_for_a'] / avg_away_goals
    stats['def_away_hist'] = stats['goals_against_a'] / avg_home_goals

    rec_att, rec_def, streaks = {}, {}, {}
    for team in stats.index:
        gf_r, ga_r, st_str = get_recent_form(df, team, n_recent)
        rec_att[team] = gf_r / avg_total
        rec_def[team] = ga_r / avg_total
        streaks[team] = st_str

    stats['recent_att'] = pd.Series(rec_att)
    stats['recent_def'] = pd.Series(rec_def)
    stats['streak'] = pd.Series(streaks)

    hw = 1.0 - form_weight
    stats['att_home'] = (hw * stats['att_home_hist']) + (form_weight * stats['recent_att'])
    stats['def_home'] = (hw * stats['def_home_hist']) + (form_weight * stats['recent_def'])
    stats['att_away'] = (hw * stats['att_away_hist']) + (form_weight * stats['recent_att'])
    stats['def_away'] = (hw * stats['def_away_hist']) + (form_weight * stats['recent_def'])

    return stats, avg_home_goals, avg_away_goals

def predict_match_pro(home_team, away_team, stats, elo_dict, avg_h, avg_a, max_goals=8):
    base_lh = stats.loc[home_team, 'att_home'] * stats.loc[away_team, 'def_away'] * avg_h
    base_la = stats.loc[away_team, 'att_away'] * stats.loc[home_team, 'def_home'] * avg_a

    elo_diff = (elo_dict[home_team] - elo_dict[away_team]) / 400.0
    lambda_home = max(0.15, base_lh * (10 ** (elo_diff * 0.10)))
    lambda_away = max(0.15, base_la * (10 ** (-elo_diff * 0.10)))

    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals)]
    matrix = np.outer(home_probs, away_probs)

    p_h = np.sum(np.tril(matrix, -1)) * 100
    p_d = np.sum(np.diag(matrix)) * 100
    p_a = np.sum(np.triu(matrix, 1)) * 100
    p_o25 = sum(matrix[i, j] for i in range(max_goals) for j in range(max_goals) if (i + j) >= 3) * 100
    p_btts = sum(matrix[i, j] for i in range(1, max_goals) for j in range(1, max_goals)) * 100

    best_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    best_score = f"{best_idx[0]}-{best_idx[1]} ({matrix[best_idx]*100:.1f}%)"

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "prob_home": p_h,
        "prob_draw": p_d,
        "prob_away": p_a,
        "prob_over_25": p_o25,
        "prob_btts_yes": p_btts,
        "best_score": best_score,
        "matrix": matrix
    }

# --- 6. INTERFACE UTILISATEUR ---
st.title("⚽ Logiciel de Prédiction Pro v4 : Live Odds API + Combinés + Bankroll")

with st.sidebar:
    st.header("🔑 Connexion API Temps Réel")
    odds_api_key = st.text_input(
        "Clé The-Odds-API (Direct)",
        value=DEFAULT_API_KEY,
        type="password",
        help="Obtenez une clé gratuite sur the-odds-api.com (500 appels/mois) pour récupérer les matchs et les meilleures cotes en direct."
    )
    if st.button("🔄 Rafraîchir les cotes en direct"):
        st.cache_data.clear()

    st.markdown("---")
    st.header("⚙️ Paramètres du Modèle")
    selected_league = st.selectbox("Championnat", list(LEAGUES.keys()))
    selected_depth = st.selectbox("Profondeur de l'historique", list(SEASONS.keys()))
    form_weight_pct = st.slider("Poids des 5 derniers matchs (%)", 0, 80, 35, 5)
    k_factor = st.slider("Réactivité Elo (Facteur K)", 10, 40, 20, 2)
    
    st.markdown("---")
    st.header("💰 Gestion de Bankroll")
    bankroll = st.number_input("Capital total (Bankroll en €)", min_value=10.0, value=500.0, step=50.0)
    combo_size = st.radio("Taille du Ticket Combiné Auto", [2, 3], index=0, horizontal=True)

league_info = LEAGUES[selected_league]
df_history = load_historical_data(league_info["code"], SEASONS[selected_depth])

if df_history.empty:
    st.error("Impossible de charger les données historiques. Vérifiez votre connexion internet.")
else:
    elo_ratings = calculate_elo_ratings(df_history, k_factor=k_factor)
    stats, avg_h, avg_a = calculate_weighted_strengths(df_history, form_weight=form_weight_pct/100.0, n_recent=5)
    valid_teams = sorted(df_history['HomeTeam'].unique())

    df_upcoming, api_status = fetch_upcoming_matches(
        league_info["code"], league_info["odds_api"], odds_api_key, valid_teams, df_history
    )

    if api_status:
        st.info(api_status)

    tab_scanner, tab_combo, tab_sim, tab_hist = st.tabs([
        "⚡ Scanner Matchs & Cotes en Direct",
        "🎟️ Générateur de Combinés (Duo/Trio) & Spécial '2'",
        "🔍 Simulateur Match par Match",
        "📜 Base Historique & Classement Elo"
    ])

    # Pré-calcul de tous les futurs matchs
    predictions_list = []
    if not df_upcoming.empty:
        for _, row in df_upcoming.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            pred = predict_match_pro(h, a, stats, elo_ratings, avg_h, avg_a)

            probs_map = {
                "1 (Domicile)": (pred['prob_home'], row['Odd_1']),
                "N (Nul)": (pred['prob_draw'], row['Odd_N']),
                "2 (Extérieur)": (pred['prob_away'], row['Odd_2'])
            }
            best_pick = max(probs_map.keys(), key=lambda k: probs_map[k][0])
            best_prob, best_odd = probs_map[best_pick]
            fair_odd = round(100.0 / best_prob, 2)
            effective_odd = best_odd if pd.notna(best_odd) else fair_odd

            # Détection de Value Bet sur les 3 issues
            value_str = "Cotes en attente"
            max_ev = -1.0
            if pd.notna(row['Odd_1']) and pd.notna(row['Odd_N']) and pd.notna(row['Odd_2']):
                evs = [
                    ((pred['prob_home']/100.0 * row['Odd_1']) - 1, f"1 @{row['Odd_1']}"),
                    ((pred['prob_draw']/100.0 * row['Odd_N']) - 1, f"N @{row['Odd_N']}"),
                    ((pred['prob_away']/100.0 * row['Odd_2']) - 1, f"2 @{row['Odd_2']}")
                ]
                best_ev_item = max(evs, key=lambda x: x[0])
                max_ev = best_ev_item[0]
                if max_ev > 0.02:
                    value_str = f"✅ VALUE {best_ev_item[1]} (+{max_ev*100:.1f}%)"
                else:
                    value_str = "Pas de Value"

            predictions_list.append({
                "Date": f"{row['Date']} {row['Heure']}".strip(),
                "Domicile": h,
                "Extérieur": a,
                "1 (%)": round(pred['prob_home'], 1),
                "N (%)": round(pred['prob_draw'], 1),
                "2 (%)": round(pred['prob_away'], 1),
                "Cote 1": row['Odd_1'],
                "Cote N": row['Odd_N'],
                "Cote 2": row['Odd_2'],
                "+2.5 Buts (%)": round(pred['prob_over_25'], 1),
                "Score Exact": pred['best_score'],
                "Pronostic Conseillé": f"{best_pick} ({best_prob:.1f}%)",
                "Value Bet Live": value_str,
                "_best_pick": best_pick,
                "_best_prob": best_prob,
                "_effective_odd": effective_odd,
                "_prob_2": pred['prob_away']
            })

    df_preds = pd.DataFrame(predictions_list)

    # --- ONGLET 1 : SCANNER TEMPS RÉEL ---
    with tab_scanner:
        st.subheader(f"📅 Prochains Matchs & Cotes en Direct — {selected_league}")
        if df_preds.empty:
            st.warning("Aucun futur match trouvé.")
        else:
            st.caption(f"Source active : **{df_upcoming['Source'].iloc[0]}** ({len(df_preds)} matchs analysés)")
            display_cols = [
                "Date", "Domicile", "Extérieur", "1 (%)", "N (%)", "2 (%)",
                "Cote 1", "Cote N", "Cote 2", "+2.5 Buts (%)", "Score Exact",
                "Pronostic Conseillé", "Value Bet Live"
            ]
            st.dataframe(
                df_preds.sort_values("_best_prob", ascending=False)[display_cols].style.background_gradient(
                    subset=["1 (%)", "2 (%)", "+2.5 Buts (%)"], cmap="YlGn"
                ),
                use_container_width=True,
                hide_index=True
            )

    # --- ONGLET 2 : GÉNÉRATEUR DE COMBINÉS (2 OU 3 MATCHS) & FILTRE SPÉCIAL "2" ---
    with tab_combo:
        if df_preds.empty:
            st.warning("Aucun match disponible pour générer un combiné.")
        else:
            st.subheader(f"🎟️ Ticket Combiné Optimal ({combo_size} sélections les plus fiables)")
            top_picks = df_preds.sort_values("_best_prob", ascending=False).head(combo_size)

            combined_prob = np.prod(top_picks["_best_prob"] / 100.0) * 100.0
            combined_odd = np.prod(top_picks["_effective_odd"])
            recommended_stake = round(bankroll * 0.02, 2)  # Mise prudente de 2% de la bankroll
            potential_return = round(recommended_stake * combined_odd, 2)

            for idx, r in top_picks.iterrows():
                st.markdown(
                    f"- **{r['Domicile']} vs {r['Extérieur']}** ({r['Date']}) 👉 "
                    f"**{r['Pronostic Conseillé']}** | Cote : **`{r['_effective_odd']:.2f}`**"
                )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Probabilité Réelle du Combiné", f"{combined_prob:.1f}%")
            k2.metric("Cote Totale Combinée", f"{combined_odd:.2f}")
            k3.metric("Mise Conseillée (2% Bankroll)", f"{recommended_stake:.2f} €")
            k4.metric("Gain Potentiel Brut", f"{potential_return:.2f} €", f"+{potential_return - recommended_stake:.2f} € net")

            st.markdown("---")
            st.subheader("🚀 Filtre Spécial « Cote 2 » (Meilleures Victoires à l'Extérieur)")
            away_favs = df_preds.sort_values("_prob_2", ascending=False).head(5)
            st.dataframe(
                away_favs[["Date", "Domicile", "Extérieur", "2 (%)", "Cote 2", "Score Exact", "Value Bet Live"]],
                use_container_width=True,
                hide_index=True
            )

    # --- ONGLET 3 : SIMULATEUR DÉTAILLÉ ---
    with tab_sim:
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Équipe à Domicile", valid_teams, index=0)
            st.caption(f"**Elo :** `{elo_ratings[home_team]:.0f}` | **5 derniers :** {stats.loc[home_team, 'streak']}")
        with col2:
            away_team = st.selectbox("✈️ Équipe à l'Extérieur", valid_teams, index=min(1, len(valid_teams)-1))
            st.caption(f"**Elo :** `{elo_ratings[away_team]:.0f}` | **5 derniers :** {stats.loc[away_team, 'streak']}")

        if home_team != away_team:
            pred = predict_match_pro(home_team, away_team, stats, elo_ratings, avg_h, avg_a)
            score_df = pd.DataFrame(
                pred['matrix'][:6, :6] * 100,
                index=[f"{home_team} {i}" for i in range(6)],
                columns=[f"{away_team} {i}" for i in range(6)]
            )
            st.subheader("🎯 Matrice des Scores Exacts (0 à 5 buts)")
            st.dataframe(score_df.style.format("{:.2f}%").background_gradient(cmap="Greens"))

    # --- ONGLET 4 : HISTORIQUE & ELO ---
    with tab_hist:
        col_elo, col_hist = st.columns([1, 2])
        with col_elo:
            st.subheader("🏆 Classement Elo")
            elo_table = pd.DataFrame({
                "Équipe": list(elo_ratings.keys()),
                "Elo": [round(v, 1) for v in elo_ratings.values()],
                "Forme (5m)": [stats.loc[t, 'streak'] for t in elo_ratings.keys()]
            }).sort_values("Elo", ascending=False).reset_index(drop=True)
            st.dataframe(elo_table, use_container_width=True)
        with col_hist:
            st.subheader(f"📜 Historique Complet ({len(df_history)} matchs)")
            st.dataframe(df_history[['Date', 'Season', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam', 'FTR']].iloc[::-1], use_container_width=True)
