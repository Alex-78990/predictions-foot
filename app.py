import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
import requests
from concurrent.futures import ThreadPoolExecutor
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
def build_predictions(df_upcoming, stats, elo_ratings, avg_h, avg_a):
    predictions_list = []
    if df_upcoming.empty:
        return predictions_list
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
    return predictions_list


CUPS = {
    "Ligue des Champions": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "Conference League": "soccer_uefa_europa_conference_league"
}

# Écart de niveau estimé entre championnats (points Elo ajoutés), pour comparer des clubs de pays différents.
LEAGUE_ELO_OFFSET = {
    "Premier League (Angleterre)": 60,
    "La Liga (Espagne)": 40,
    "Bundesliga (Allemagne)": 20,
    "Serie A (Italie)": 15,
    "Ligue 1 (France)": 0
}

ALL_COMPETITIONS = list(LEAGUES.keys()) + list(CUPS.keys())


def read_api_keys_from_secrets():
    """Clés enregistrées côté serveur : ODDS_API_KEY (une clé) et ODDS_API_KEYS (liste de clés)."""
    try:
        single = st.secrets.get("ODDS_API_KEY", "")
        multi = st.secrets.get("ODDS_API_KEYS", [])
    except Exception:
        return []
    if isinstance(multi, str):
        multi = multi.replace(",", "\n").splitlines()
    keys = []
    for k in [single, *list(multi)]:
        k = str(k).strip()
        if len(k) > 10 and k not in keys:
            keys.append(k)
    return keys


def match_cup_team(name, valid_teams):
    """Rapprochement strict des noms d'équipes pour les coupes (évite les faux positifs sur les clubs étrangers)."""
    clean = TEAM_ALIASES.get(name, name)
    if clean in valid_teams:
        return clean
    lowered = {t.lower(): t for t in valid_teams}
    if clean.lower() in lowered:
        return lowered[clean.lower()]
    contained = [t for t in valid_teams if len(t) >= 6 and t.lower() in clean.lower()]
    if len(contained) == 1:
        return contained[0]
    close = get_close_matches(clean, valid_teams, n=1, cutoff=0.8)
    return close[0] if close else None


def fetch_league_with_rotation(info, api_keys, valid_teams, df_history):
    """Essaie les clés dans l'ordre : passe à la suivante si la clé est invalide, expirée ou épuisée."""
    keys = list(api_keys) if api_keys else [""]
    result = (pd.DataFrame(), None)
    for pos, key in enumerate(keys):
        df_up, status = fetch_upcoming_matches(info["code"], info["odds_api"], key, valid_teams, df_history)
        if pos > 0 and status and status.startswith("🟢"):
            status += f" | ↪ clé n°{pos + 1} utilisée (les précédentes sont expirées ou épuisées)"
        result = (df_up, status)
        if not status or not status.startswith(("🔴", "🟠")):
            break
    return result


def parse_cup_events(events, valid_teams):
    rows, skipped = [], 0
    for item in events:
        raw_h, raw_a = item.get('home_team', ''), item.get('away_team', '')
        h, a = match_cup_team(raw_h, valid_teams), match_cup_team(raw_a, valid_teams)
        if not h or not a or h == a:
            skipped += 1
            continue
        odds_1, odds_n, odds_2 = [], [], []
        for bm in item.get('bookmakers', []):
            for market in bm.get('markets', []):
                if market.get('key') != 'h2h':
                    continue
                for out in market.get('outcomes', []):
                    if out['name'] == raw_h:
                        odds_1.append(float(out['price']))
                    elif out['name'] == raw_a:
                        odds_2.append(float(out['price']))
                    else:
                        odds_n.append(float(out['price']))
        commence = item.get('commence_time', '')
        rows.append({
            "Date": commence[:10] if commence else "À venir",
            "Heure": commence[11:16] if len(commence) >= 16 else "",
            "HomeTeam": h,
            "AwayTeam": a,
            "Odd_1": round(max(odds_1), 2) if odds_1 else np.nan,
            "Odd_N": round(max(odds_n), 2) if odds_n else np.nan,
            "Odd_2": round(max(odds_2), 2) if odds_2 else np.nan,
            "Bookmakers": len(item.get('bookmakers', [])),
            "Source": "⚡ The Odds API (Temps Réel)"
        })
    return rows, skipped


@st.cache_data(ttl=300, show_spinner=False)
def fetch_cup_matches(sport_key, api_keys, valid_teams):
    """Matchs de coupes d'Europe via The Odds API, avec bascule sur les clés suivantes."""
    status, skipped = None, 0
    for pos, key in enumerate(api_keys):
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
            f"?apiKey={key.strip()}&regions=eu&markets=h2h&oddsFormat=decimal"
        )
        try:
            res = requests.get(url, timeout=10)
        except Exception as e:
            status = f"🟠 Coupes : erreur réseau API ({str(e)[:40]})"
            continue
        rem = res.headers.get("x-requests-remaining", "?")
        if res.status_code == 200:
            rows, skipped = parse_cup_events(res.json(), valid_teams)
            status = f"🟢 Coupes d'Europe connectées en direct | clé n°{pos + 1} | requêtes restantes : {rem}"
            return pd.DataFrame(rows), status, skipped
        if res.status_code in (401, 429):
            status = f"🔴 Coupes : clé n°{pos + 1} invalide ou quota atteint (erreur {res.status_code})"
            continue
        status = f"🟠 Coupes : compétition indisponible (erreur {res.status_code})"
        break
    if not api_keys:
        status = "🟠 Coupes d'Europe : une clé The Odds API est nécessaire."
    return pd.DataFrame(), status, skipped


@st.cache_data(ttl=3600, show_spinner=False)
def build_league_model(league_name, season_key, form_weight_pct, k_factor):
    """Historique, Elo et forces d'attaque/défense d'un championnat (sans appel API)."""
    info = LEAGUES[league_name]
    df_history = load_historical_data(info["code"], SEASONS[season_key])
    if df_history.empty:
        return None
    elo_ratings = calculate_elo_ratings(df_history, k_factor=k_factor)
    stats, avg_h, avg_a = calculate_weighted_strengths(df_history, form_weight=form_weight_pct/100.0, n_recent=5)
    return {
        "df_history": df_history, "elo": elo_ratings, "stats": stats,
        "avg_h": avg_h, "avg_a": avg_a, "valid_teams": sorted(df_history['HomeTeam'].unique())
    }


@st.cache_data(ttl=3600, show_spinner=False)
def build_european_model(season_key, form_weight_pct, k_factor):
    """Modèle commun aux 5 championnats pour comparer les clubs en coupe d'Europe."""
    stats_frames, elo, avg_h_list, avg_a_list = [], {}, [], []
    for name in LEAGUES:
        model = build_league_model(name, season_key, form_weight_pct, k_factor)
        if model is None:
            continue
        stats_frames.append(model["stats"])
        for team, value in model["elo"].items():
            elo[team] = value + LEAGUE_ELO_OFFSET[name]
        avg_h_list.append(model["avg_h"])
        avg_a_list.append(model["avg_a"])
    if not stats_frames:
        return None
    stats = pd.concat(stats_frames)
    stats = stats[~stats.index.duplicated(keep="last")]
    return {
        "stats": stats, "elo": elo, "teams": sorted(stats.index),
        "avg_h": float(np.mean(avg_h_list)), "avg_a": float(np.mean(avg_a_list))
    }


@st.cache_data(ttl=300, show_spinner=False)
def build_league_bundle(league_name, season_key, form_weight_pct, k_factor, api_keys):
    """Modèle + pronostics d'un championnat."""
    model = build_league_model(league_name, season_key, form_weight_pct, k_factor)
    if model is None:
        return None
    df_upcoming, api_status = fetch_league_with_rotation(
        LEAGUES[league_name], api_keys, model["valid_teams"], model["df_history"]
    )
    df_preds = pd.DataFrame(build_predictions(df_upcoming, model["stats"], model["elo"], model["avg_h"], model["avg_a"]))
    if not df_preds.empty:
        df_preds.insert(0, "Championnat", league_name)
    source = df_upcoming['Source'].iloc[0] if not df_upcoming.empty else "Aucune source"
    return {**model, "df_preds": df_preds, "api_status": api_status, "source": source, "skipped": 0}


def build_cup_bundle(cup_name, season_key, form_weight_pct, k_factor, api_keys):
    """Pronostics d'une coupe d'Europe (clubs des 5 grands championnats uniquement)."""
    model = build_european_model(season_key, form_weight_pct, k_factor)
    if model is None:
        return None
    df_cup, status, skipped = fetch_cup_matches(CUPS[cup_name], api_keys, model["teams"])
    df_preds = pd.DataFrame(build_predictions(df_cup, model["stats"], model["elo"], model["avg_h"], model["avg_a"]))
    if not df_preds.empty:
        df_preds.insert(0, "Championnat", cup_name)
    source = "⚡ The Odds API (Temps Réel)" if not df_cup.empty else "Aucun match"
    return {"df_preds": df_preds, "api_status": status, "source": source, "skipped": skipped}


ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def odds_api_get(path, params, api_keys):
    """GET sur The Odds API en essayant les clés dans l'ordre (401 ou 429 : clé suivante)."""
    status = None
    for pos, key in enumerate(api_keys):
        try:
            res = requests.get(f"{ODDS_API_BASE}{path}", params={**params, "apiKey": key.strip()}, timeout=12)
        except Exception as e:
            status = f"🟠 Erreur réseau API ({str(e)[:40]})"
            continue
        if res.status_code == 200:
            remaining = res.headers.get("x-requests-remaining", "?")
            return res.json(), f"🟢 Clé n°{pos + 1} | requêtes restantes : {remaining}"
        if res.status_code in (401, 429):
            status = f"🔴 Clé n°{pos + 1} invalide ou quota atteint (erreur {res.status_code})"
            continue
        status = f"🟠 Erreur API {res.status_code}"
        break
    return None, status or "🟠 Aucune clé API disponible."


@st.cache_data(ttl=3600, show_spinner=False)
def list_active_sports(api_keys):
    """Liste des compétitions actives (cet appel ne consomme pas de quota)."""
    data, status = odds_api_get("/sports/", {}, api_keys)
    if not data:
        return [], status
    sports = [s for s in data if s.get("active") and not s.get("has_outrights")]
    return sorted(sports, key=lambda s: (s["group"], s["title"])), status


def consensus_rows(events, sport_label):
    """Probabilités de consensus (marge retirée) et meilleures cotes pour chaque match."""
    rows = []
    for ev in events:
        home, away = ev.get("home_team", ""), ev.get("away_team", "")
        probs, best, n_books = {}, {}, 0
        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: float(o["price"]) for o in market.get("outcomes", []) if float(o["price"]) > 1}
                if len(prices) < 2:
                    continue
                inverse = {n: 1.0 / p for n, p in prices.items()}
                total = sum(inverse.values())
                n_books += 1
                for n, p in prices.items():
                    probs.setdefault(n, []).append(inverse[n] / total)
                    best[n] = max(best.get(n, 0.0), p)
        if n_books == 0:
            continue
        consensus = {n: float(np.mean(v)) for n, v in probs.items()}
        favourite = max(consensus, key=consensus.get)
        value_pick = max(consensus, key=lambda n: consensus[n] * best[n])
        value_pct = (consensus[value_pick] * best[value_pick] - 1) * 100
        is_value = value_pct > 2.0 and n_books >= 3
        rows.append({
            "Sport": sport_label,
            "_kickoff": ev.get("commence_time", ""),
            "Match": f"{home} vs {away}",
            "Favori": favourite,
            "Proba favori (%)": round(consensus[favourite] * 100, 1),
            "Meilleure cote favori": round(best[favourite], 2),
            "Meilleure value": f"{value_pick} @{best[value_pick]:.2f}",
            "Écart vs consensus (%)": round(value_pct, 1),
            "Bookmakers": n_books,
            "Value": "✅ VALUE" if is_value else "Pas de value"
        })
    return rows


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sport_odds(sport_key, sport_label, api_keys):
    data, status = odds_api_get(
        f"/sports/{sport_key}/odds/",
        {"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
        api_keys
    )
    return (consensus_rows(data, sport_label) if data else []), status


@st.cache_data(ttl=600, show_spinner=False)
def fetch_events_by_sport(sport_keys, api_keys):
    """Heures de coup d'envoi des matchs à venir par compétition (appel gratuit, sans quota)."""
    def one(key):
        data, _ = odds_api_get(f"/sports/{key}/events", {}, api_keys)
        return key, [e.get("commence_time", "") for e in (data or [])]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, sport_keys))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_many_sport_odds(items, api_keys):
    """items : tuple de (clé, libellé). Retourne les lignes de consensus et les statuts API."""
    def one(item):
        key, label = item
        data, status = odds_api_get(
            f"/sports/{key}/odds/",
            {"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
            api_keys
        )
        return (consensus_rows(data, label) if data else []), status
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, items))
    rows = [row for res in results for row in res[0]]
    statuses = [res[1] for res in results if res[1]]
    return rows, statuses


st.title("⚽ Logiciel de Prédiction Pro v4 : Championnats + Coupes d'Europe + Combinés + Bankroll")

secret_keys = read_api_keys_from_secrets()

with st.sidebar:
    st.header("🔑 Connexion API Temps Réel")
    if secret_keys:
        st.caption(f"🔒 {len(secret_keys)} clé(s) enregistrée(s) côté serveur (non affichée(s)).")
    st.caption("Clés de secours : utilisées dans l'ordre quand la précédente est invalide, expirée ou épuisée.")
    extra_keys = [
        st.text_input(
            f"Clé The-Odds-API n°{i}", type="password", key=f"extra_key_{i}",
            help="Clé gratuite sur the-odds-api.com (500 appels/mois)."
        ) for i in (1, 2, 3)
    ]
    api_keys = tuple(dict.fromkeys(
        [*secret_keys, *[k.strip() for k in extra_keys if len(k.strip()) > 10]]
    ))
    if st.button("🔄 Rafraîchir les cotes en direct"):
        st.cache_data.clear()

    st.markdown("---")
    st.header("⚙️ Paramètres du Modèle")
    competitions = st.multiselect(
        "Compétitions à analyser",
        ALL_COMPETITIONS,
        default=ALL_COMPETITIONS,
        help="Chaque compétition coûte 1 requête API, mise en cache 5 minutes."
    )
    selected_league = st.selectbox("Championnat (simulateur et historique)", list(LEAGUES.keys()))
    selected_depth = st.selectbox("Profondeur de l'historique", list(SEASONS.keys()))
    form_weight_pct = st.slider("Poids des 5 derniers matchs (%)", 0, 80, 35, 5)
    k_factor = st.slider("Réactivité Elo (Facteur K)", 10, 40, 20, 2)

    st.markdown("---")
    st.header("💰 Gestion de Bankroll")
    bankroll = st.number_input("Capital total (Bankroll en €)", min_value=10.0, value=500.0, step=50.0)
    combo_size = st.radio("Taille du Ticket Combiné Auto", [2, 3], index=0, horizontal=True)

sel = build_league_bundle(selected_league, selected_depth, form_weight_pct, k_factor, api_keys)

if sel is None:
    st.error("Impossible de charger les données historiques. Vérifiez votre connexion internet.")
else:
    scan_results = {}
    with st.spinner("Analyse des compétitions en cours..."):
        for name in competitions:
            if name in LEAGUES:
                bundle = build_league_bundle(name, selected_depth, form_weight_pct, k_factor, api_keys)
            else:
                bundle = build_cup_bundle(name, selected_depth, form_weight_pct, k_factor, api_keys)
            if bundle is not None:
                scan_results[name] = bundle

    df_history, elo_ratings, stats = sel["df_history"], sel["elo"], sel["stats"]
    avg_h, avg_a, valid_teams = sel["avg_h"], sel["avg_a"], sel["valid_teams"]

    frames = [b["df_preds"] for b in scan_results.values() if not b["df_preds"].empty]
    df_preds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    statuses = [(n, b["api_status"]) for n, b in scan_results.items() if b["api_status"]]
    if statuses:
        st.info(statuses[-1][1])
        for n, msg in statuses:
            if msg.startswith("🔴") or msg.startswith("🟠"):
                st.warning(f"{n} : {msg}")

    tab_next, tab_scanner, tab_combo, tab_sim, tab_hist, tab_sports = st.tabs([
        "⏱️ Prochaines 24h",
        "⚡ Scanner Matchs & Cotes en Direct",
        "🎟️ Générateur de Combinés (Duo/Trio) & Spécial '2'",
        "🔍 Simulateur Match par Match",
        "📜 Base Historique & Classement Elo",
        "🌍 Autres sports (cotes)"
    ])

    # --- ONGLET 0 : PROCHAINS MATCHS (TOUTES COMPÉTITIONS, FENÊTRE EN HEURES) ---
    with tab_next:
        window_h = st.slider("Fenêtre à venir (heures)", 6, 72, 24, 6)
        if df_preds.empty:
            st.warning("Aucun match disponible pour les compétitions choisies.")
        else:
            date_text = df_preds["Date"].astype(str).str.strip()
            kickoff = pd.to_datetime(date_text, format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
            kickoff = kickoff.fillna(pd.to_datetime(date_text, format="%d/%m/%Y %H:%M", errors="coerce", utc=True))
            kickoff = kickoff.fillna(pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce", utc=True))
            now_utc = pd.Timestamp.now(tz="UTC")
            in_window = (kickoff >= now_utc - pd.Timedelta(hours=2)) & (kickoff <= now_utc + pd.Timedelta(hours=window_h))
            soon = df_preds[in_window].copy()
            soon["_kickoff"] = kickoff[in_window]
            soon = soon.sort_values("_kickoff")
            soon["Coup d'envoi (Paris)"] = soon["_kickoff"].dt.tz_convert("Europe/Paris").dt.strftime("%a %d/%m %H:%M")
            st.subheader(f"⏱️ Matchs des {window_h} prochaines heures — toutes compétitions")
            st.caption(
                f"{len(soon)} match(s) sur {soon['Championnat'].nunique() if not soon.empty else 0} compétition(s). "
                "Heures affichées à l'heure de Paris, matchs commencés depuis moins de 2 h inclus."
            )
            if soon.empty:
                st.info("Aucun match dans cette fenêtre. Essayez 48 ou 72 heures.")
            else:
                n_value = int(soon["Value Bet Live"].str.contains("VALUE").sum())
                m1, m2 = st.columns(2)
                m1.metric("Matchs à venir", len(soon))
                m2.metric("Value Bets détectés", n_value)
                next_cols = [
                    "Coup d'envoi (Paris)", "Championnat", "Domicile", "Extérieur",
                    "1 (%)", "N (%)", "2 (%)", "Cote 1", "Cote N", "Cote 2",
                    "Pronostic Conseillé", "Value Bet Live"
                ]
                st.dataframe(
                    soon[next_cols].style.background_gradient(subset=["1 (%)", "2 (%)"], cmap="YlGn"),
                    use_container_width=True,
                    hide_index=True
                )

    # --- ONGLET 1 : SCANNER TEMPS RÉEL (CHAMPIONNATS + COUPES) ---
    with tab_scanner:
        st.subheader(f"📅 Prochains Matchs & Cotes en Direct — {len(competitions)} compétition(s)")
        if df_preds.empty:
            st.warning("Aucun futur match trouvé pour les compétitions choisies.")
        else:
            sources = sorted({b["source"] for b in scan_results.values() if not b["df_preds"].empty})
            st.caption(f"Source active : **{' / '.join(sources)}** ({len(df_preds)} matchs analysés sur {len(frames)} compétition(s))")
            cups_in_view = [n for n in scan_results if n in CUPS and not scan_results[n]["df_preds"].empty]
            if cups_in_view:
                skipped_total = sum(scan_results[n]["skipped"] for n in cups_in_view)
                st.caption(
                    "Coupes d'Europe : seuls les clubs des 5 grands championnats sont analysés "
                    f"({skipped_total} match(s) ignoré(s)). Le niveau relatif des championnats est estimé "
                    "(Premier League +60, Liga +40, Bundesliga +20, Serie A +15, Ligue 1 +0 points Elo)."
                )
            shown_leagues = st.multiselect("Compétitions affichées", list(df_preds["Championnat"].unique()), default=list(df_preds["Championnat"].unique()))
            view = df_preds[df_preds["Championnat"].isin(shown_leagues)]
            only_value = st.checkbox("Afficher seulement les Value Bets")
            if only_value:
                view = view[view["Value Bet Live"].str.contains("VALUE")]
            display_cols = [
                "Championnat", "Date", "Domicile", "Extérieur", "1 (%)", "N (%)", "2 (%)",
                "Cote 1", "Cote N", "Cote 2", "+2.5 Buts (%)", "Score Exact",
                "Pronostic Conseillé", "Value Bet Live"
            ]
            if view.empty:
                st.info("Aucun match ne correspond à ce filtre.")
            else:
                st.dataframe(
                    view.sort_values("_best_prob", ascending=False)[display_cols].style.background_gradient(
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
                    f"- [{r['Championnat']}] **{r['Domicile']} vs {r['Extérieur']}** ({r['Date']}) 👉 "
                    f"**{r['Pronostic Conseillé']}** | Cote : **{r['_effective_odd']:.2f}**"
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
                away_favs[["Championnat", "Date", "Domicile", "Extérieur", "2 (%)", "Cote 2", "Score Exact", "Value Bet Live"]],
                use_container_width=True,
                hide_index=True
            )

    # --- ONGLET 3 : SIMULATEUR DÉTAILLÉ (CHAMPIONNAT SÉLECTIONNÉ) ---
    with tab_sim:
        st.caption(f"Équipes de : {selected_league}")
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("🏠 Équipe à Domicile", valid_teams, index=0)
            st.caption(f"Elo : {elo_ratings[home_team]:.0f} | 5 derniers : {stats.loc[home_team, 'streak']}")
        with col2:
            away_team = st.selectbox("✈️ Équipe à l'Extérieur", valid_teams, index=min(1, len(valid_teams)-1))
            st.caption(f"Elo : {elo_ratings[away_team]:.0f} | 5 derniers : {stats.loc[away_team, 'streak']}")

        if home_team != away_team:
            pred = predict_match_pro(home_team, away_team, stats, elo_ratings, avg_h, avg_a)
            score_df = pd.DataFrame(
                pred['matrix'][:6, :6] * 100,
                index=[f"{home_team} {i}" for i in range(6)],
                columns=[f"{away_team} {i}" for i in range(6)]
            )
            st.subheader("🎯 Matrice des Scores Exacts (0 à 5 buts)")
            st.dataframe(score_df.style.format("{:.2f}%").background_gradient(cmap="Greens"))

    # --- ONGLET 4 : HISTORIQUE & ELO (CHAMPIONNAT SÉLECTIONNÉ) ---
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


    # --- ONGLET 5 : AUTRES SPORTS (TOUS LES MATCHS DU CRÉNEAU, COTES COMPARÉES) ---
    with tab_sports:
        st.subheader("🌍 Autres sports : tous les matchs du créneau, cotes comparées")
        st.caption(
            "Pour ces sports, il n'y a pas d'historique de résultats : les probabilités sont le consensus des bookmakers "
            "(marge retirée). « Value » signale une meilleure cote supérieure de plus de 2 % à la cote juste du consensus, "
            "avec au moins 3 bookmakers. Les compétitions sans match dans le créneau ne sont pas interrogées."
        )
        all_sports, sports_status = list_active_sports(api_keys)
        if not all_sports:
            st.warning(sports_status)
        else:
            horizon_h = st.slider("Créneau : matchs dans les prochaines (heures)", 6, 168, 24, 6, key="sports_horizon")
            groups = sorted({s["group"] for s in all_sports})
            chosen_groups = st.multiselect("Sports", groups, default=groups)
            max_calls = st.number_input(
                "Plafond de requêtes API par chargement", min_value=1, max_value=70, value=25, step=5,
                help="Une requête par compétition ayant des matchs dans le créneau. Protège votre quota mensuel."
            )
            pool_sports = [s for s in all_sports if s["group"] in chosen_groups]
            events_by_sport = fetch_events_by_sport(tuple(s["key"] for s in pool_sports), api_keys)
            now_utc = pd.Timestamp.now(tz="UTC")
            window_start = now_utc - pd.Timedelta(hours=2)
            window_end = now_utc + pd.Timedelta(hours=horizon_h)

            def count_in_window(times):
                stamps = pd.to_datetime(pd.Series(times, dtype="object"), errors="coerce", utc=True)
                return int(((stamps >= window_start) & (stamps <= window_end)).sum())

            active = [(s, count_in_window(events_by_sport.get(s["key"], []))) for s in pool_sports]
            active = sorted([(s, n) for s, n in active if n > 0], key=lambda x: -x[1])
            if not active:
                st.info("Aucun match dans ce créneau pour les sports choisis. Élargissez le nombre d'heures.")
            else:
                to_query = active[:int(max_calls)]
                if len(active) > len(to_query):
                    st.warning(
                        f"{len(active)} compétitions ont des matchs dans le créneau : les {len(to_query)} qui en ont le plus "
                        "sont analysées (plafond de requêtes). Réduisez le créneau ou les sports, ou relevez le plafond."
                    )
                st.caption(
                    f"{sum(n for _, n in active)} match(s) annoncé(s) sur {len(active)} compétition(s). "
                    f"Requêtes API utilisées : {len(to_query)} (mises en cache 5 minutes)."
                )
                items = tuple((s["key"], f"{s['group']} · {s['title']}") for s, _ in to_query)
                with st.spinner("Récupération des cotes de tous les matchs du créneau..."):
                    sports_rows, sports_statuses = fetch_many_sport_odds(items, api_keys)
                good = [m for m in sports_statuses if m.startswith("🟢")]
                if good:
                    st.info(good[-1])
                for message in sorted({m for m in sports_statuses if m.startswith(("🔴", "🟠"))}):
                    st.warning(message)
                sports_df = pd.DataFrame(sports_rows)
                if sports_df.empty:
                    st.info("Aucun match avec cotes pour cette sélection.")
                else:
                    kick = pd.to_datetime(sports_df["_kickoff"], errors="coerce", utc=True)
                    view_df = sports_df[(kick >= window_start) & (kick <= window_end)].copy()
                    view_df["_k"] = kick[view_df.index]
                    view_df = view_df.sort_values("_k")
                    view_df["Coup d'envoi (Paris)"] = view_df["_k"].dt.tz_convert("Europe/Paris").dt.strftime("%a %d/%m %H:%M")
                    only_value_sports = st.checkbox("Afficher seulement les Value", key="sports_only_value")
                    if only_value_sports:
                        view_df = view_df[view_df["Value"] == "✅ VALUE"]
                    n_value_sports = int((view_df["Value"] == "✅ VALUE").sum()) if not view_df.empty else 0
                    c1, c2 = st.columns(2)
                    c1.metric("Matchs dans le créneau", len(view_df))
                    c2.metric("Value détectées", n_value_sports)
                    if view_df.empty:
                        st.info("Aucun match à afficher avec ce filtre.")
                    else:
                        st.dataframe(
                            view_df[[
                                "Coup d'envoi (Paris)", "Sport", "Match", "Favori", "Proba favori (%)",
                                "Meilleure cote favori", "Meilleure value", "Écart vs consensus (%)", "Bookmakers", "Value"
                            ]],
                            use_container_width=True,
                            hide_index=True
                        )
