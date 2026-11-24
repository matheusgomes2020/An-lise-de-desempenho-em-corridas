import json
from pathlib import Path
import numpy as np
import pandas as pd

caminho_padrao = Path(
    r"C:\Users\batat\Downloads\Python\POS\Regressão Linear\dados_corrida\analise_corridas\new\Análise de desempenho em corridas\Motion path detail data & description"
)

# Treinos gravados errado (ex.: bike no modo corrida) removidos dos 3 CSVs
IDS_EXCLUIR = [
    "vrh+DwAAAAACAAAA6DwriZgBAAA=",  # 2025-08-08 - cadência 7 / 104 bpm
    "bFAkEgAAAAACAAAAfhfQdp0BAAA=",  # 2026-04-10 - bike manhã / cadência 25
]

# Opcional: adicione aqui pesagens posteriores a 2026-07-10 se o peso mudou
PESO_MANUAL = [
    # ("2026-09-15", 78.5),
]


def localizar_treinos():
  pasta_alvo = caminho_padrao if caminho_padrao.exists() else None
  if pasta_alvo is None:
    for base in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
      if not base.exists():
        continue
      candidatos = [p for p in base.rglob("*Motion path*") if p.is_dir()]
      if candidatos:
        pasta_alvo = candidatos[0]
        break

  if pasta_alvo is None:
    pasta_alvo = Path.cwd()

  if (
      "todos_treinos" in globals()
      and isinstance(todos_treinos, list)
      and len(todos_treinos) > 0
  ):
    print(f"Reaproveitando {len(todos_treinos)} sessões já carregadas na memória.")
    return todos_treinos, pasta_alvo

  treinos = []
  for arq in pasta_alvo.rglob("*.json"):
    if ".ipynb_checkpoints" in str(arq):
      continue
    try:
      with open(arq, "r", encoding="utf-8") as f:
        dados = json.load(f)
        lista = dados if isinstance(dados, list) else [dados]
        treinos.extend(
            [d for d in lista if isinstance(d, dict) and "detailData" in d]
        )
    except Exception:
      continue

  print(f"Lidas {len(treinos)} sessões de: {pasta_alvo}")
  return treinos, pasta_alvo


def extrair_historico_peso(pasta_raiz):
  """Lê pesagens dos JSONs do export da Huawei e combina com PESO_MANUAL."""
  registros = []

  if pasta_raiz.exists():
    for arq in pasta_raiz.rglob("*.json"):
      if "Motion path" in str(arq) or ".ipynb_checkpoints" in str(arq):
        continue
      try:
        texto = arq.read_text(encoding="utf-8", errors="ignore")
        if "bodyWeight" not in texto and "weight" not in texto.lower():
          continue
        dados = json.loads(texto)
        itens = dados if isinstance(dados, list) else [dados]
        for item in itens:
          if not isinstance(item, dict):
            continue
          for pt in item.get("samplePoints", []):
            val_raw = pt.get("value", "")
            val = (
                json.loads(val_raw)
                if isinstance(val_raw, str) and val_raw.startswith("{")
                else val_raw
            )
            if isinstance(val, dict) and "bodyWeight" in val:
              ts = pt.get("startTime") or item.get("startTime")
              peso = float(val["bodyWeight"])
              if ts and 30 <= peso <= 200:
                registros.append((
                    (
                        pd.to_datetime(int(ts), unit="ms")
                        - pd.Timedelta(hours=3)
                    ).floor("s"),
                    round(peso, 2),
                ))
      except Exception:
        continue

  for data_str, peso_val in PESO_MANUAL:
    registros.append((pd.to_datetime(data_str), round(float(peso_val), 2)))

  if not registros:
    print("Nenhuma pesagem encontrada nos JSONs ou em PESO_MANUAL.")
    return pd.DataFrame(columns=["data_peso", "peso_kg"])

  df_peso = (
      pd.DataFrame(registros, columns=["data_peso", "peso_kg"])
      .sort_values("data_peso")
      .drop_duplicates("data_peso", keep="last")
      .reset_index(drop=True)
  )
  print(f"Pesagens carregadas: {len(df_peso)} registros.")
  return df_peso


def formatar_pace(pace_dec):
  if pd.isna(pace_dec) or pace_dec <= 0:
    return None
  total_seg = int(round(pace_dec * 60))
  minutos, segundos = divmod(total_seg, 60)
  return f"{minutos}:{segundos:02d}"


def formatar_duracao_segundos(segundos_totais):
  if pd.isna(segundos_totais) or segundos_totais <= 0:
    return None
  total_seg = int(round(segundos_totais))
  horas, resto = divmod(total_seg, 3600)
  minutos, segundos = divmod(resto, 60)
  if horas > 0:
    return f"{horas}:{minutos:02d}:{segundos:02d}"
  return f"{minutos:02d}:{segundos:02d}"


def extrair_sensores(detail_str):
  hr_rows, cad_rows, alti_rows, rs_rows = [], [], [], []

  for linha in detail_str.splitlines():
    if not linha.startswith(("tp=h-r;", "tp=s-r;", "tp=alti;", "tp=rs;")):
      continue
    p = dict(item.split("=", 1) for item in linha.split(";") if "=" in item)
    tp = p.get("tp")
    k_val = float(p.get("k", 0))
    val = float(p.get("v", 0))

    if tp == "h-r" and val > 30:
      hr_rows.append((int(k_val), val))
    elif tp == "s-r" and val >= 60:
      cad_rows.append((int(k_val), val))
    elif tp == "alti" and val > 0:
      alti_rows.append((int(k_val), val))
    elif tp == "rs" and val >= 0:
      rs_rows.append((k_val, min(val / 10.0, 6.5)))

  df_hr = (
      pd.DataFrame(hr_rows, columns=["ts_ms", "bpm"])
      .sort_values("ts_ms")
      .drop_duplicates("ts_ms")
      if hr_rows
      else pd.DataFrame(columns=["ts_ms", "bpm"])
  )
  df_cad = (
      pd.DataFrame(cad_rows, columns=["ts_ms", "cadencia"])
      .sort_values("ts_ms")
      .drop_duplicates("ts_ms")
      if cad_rows
      else pd.DataFrame(columns=["ts_ms", "cadencia"])
  )

  if alti_rows:
    df_alti = (
        pd.DataFrame(alti_rows, columns=["ts_ms", "altitude"])
        .sort_values("ts_ms")
        .drop_duplicates("ts_ms")
        .reset_index(drop=True)
    )
    df_alti["altitude_smooth"] = (
        df_alti["altitude"].rolling(3, center=True, min_periods=1).mean()
    )
  else:
    df_alti = pd.DataFrame(columns=["ts_ms", "altitude", "altitude_smooth"])

  return df_hr, df_cad, df_alti, rs_rows


def construir_janelas_km(
    part_time_map, total_dist_m, total_time_s, start_ms, end_ms, rs_rows
):
  janelas = []

  if isinstance(part_time_map, dict) and len(part_time_map) > 0:
    marcos = sorted((float(k), float(v)) for k, v in part_time_map.items())
    t_ant, dist_ant = 0.0, 0.0

    for km_float, t_acum in marcos:
      km_num = int(round(km_float))
      dist_atual = min(km_num * 1000.0, total_dist_m)
      dist_trecho = dist_atual - dist_ant
      dt_seg = t_acum - t_ant

      if dist_trecho > 20 and dt_seg > 0:
        janelas.append((
            km_num,
            dist_ant,
            dist_atual,
            dist_trecho,
            dt_seg,
            start_ms + int(t_ant * 1000),
            start_ms + int(t_acum * 1000),
            "oficial_relogio",
        ))
      t_ant = t_acum
      dist_ant = dist_atual

    dist_restante = total_dist_m - dist_ant
    dt_restante = total_time_s - t_ant
    if dist_restante >= 50 and dt_restante > 2:
      janelas.append((
          len(janelas) + 1,
          dist_ant,
          total_dist_m,
          dist_restante,
          dt_restante,
          start_ms + int(t_ant * 1000),
          start_ms + int(total_time_s * 1000),
          "oficial_relogio",
      ))
    return janelas

  if len(rs_rows) < 5:
    return []

  df_vel = pd.DataFrame(rs_rows, columns=["k_raw", "vel_ms"])
  if (df_vel["k_raw"] > 1e11).all():
    df_vel["ts_ms"] = df_vel["k_raw"].astype("int64")
    df_vel = (
        df_vel.sort_values("ts_ms")
        .drop_duplicates("ts_ms")
        .reset_index(drop=True)
    )
  else:
    df_vel = df_vel.sort_values("k_raw").reset_index(drop=True)
    df_vel["ts_ms"] = np.linspace(
        start_ms, end_ms, len(df_vel), dtype="int64"
    )

  dt = (
      df_vel["ts_ms"]
      .diff()
      .div(1000.0)
      .fillna(total_time_s / len(df_vel))
      .clip(0.5, 30.0)
  )
  dist_bruta = np.cumsum(df_vel["vel_ms"] * dt)
  if dist_bruta.iloc[-1] <= 0:
    return []

  df_vel["dist_acum_m"] = dist_bruta * (total_dist_m / dist_bruta.iloc[-1])
  df_vel["km"] = np.maximum(
      1, np.ceil(df_vel["dist_acum_m"] / 1000.0).astype(int)
  )

  dist_ant = 0.0
  ts_ant = start_ms
  for km, grp in df_vel.groupby("km"):
    dist_atual = float(grp["dist_acum_m"].iloc[-1])
    dist_trecho = dist_atual - dist_ant
    ts_atual = int(grp["ts_ms"].iloc[-1])
    dt_seg = (ts_atual - ts_ant) / 1000.0
    if dist_trecho >= 100 and dt_seg > 0:
      janelas.append((
          int(km),
          dist_ant,
          dist_atual,
          dist_trecho,
          dt_seg,
          ts_ant,
          ts_atual,
          "estimado_sensor",
      ))
    dist_ant = dist_atual
    ts_ant = ts_atual

  return janelas


def processar_tudo(treino):
  rec_id = treino.get("recordId")
  if rec_id in IDS_EXCLUIR:
    return None, None, None

  s_raw = treino.get("summaryData", "{}")
  s = json.loads(s_raw) if isinstance(s_raw, str) else (s_raw or {})
  sport_type = s.get("sportType")
  total_dist_m = float(s.get("totalDistance", 0))
  total_time_s = float(s.get("totalTime", 0)) / 1000.0
  start_ms = int(treino["startTime"])
  end_ms = int(treino.get("endTime", start_ms + int(total_time_s * 1000)))

  if sport_type not in (258, 264) or total_dist_m < 1000 or total_time_s <= 0:
    return None, None, None

  df_hr, df_cad, df_alti, rs_rows = extrair_sensores(
      treino.get("detailData", "")
  )
  janelas = construir_janelas_km(
      s.get("partTimeMap"),
      total_dist_m,
      total_time_s,
      start_ms,
      end_ms,
      rs_rows,
  )
  if not janelas:
    return None, None, None

  # Padroniza timestamp sem milissegundos para evitar erro de leitura nos CSVs
  data_treino = (
      pd.to_datetime(start_ms, unit="ms") - pd.Timedelta(hours=3)
  ).floor("s")

  modalidade = "Rua" if sport_type == 258 else "Esteira"
  dist_km = total_dist_m / 1000.0
  duracao_min = total_time_s / 60.0
  pace_treino_dec = (
      (s.get("avgPace", 0) / 60.0)
      if s.get("avgPace", 0) > 0
      else (duracao_min / dist_km)
  )
  bpm_treino_med = s.get("avgHeartRate")

  # --- 1. DF POR KM ---
  splits_km = []
  for km, d_ini, d_fim, dist_trecho, tempo_seg, ts_ini, ts_fim, origem in janelas:
    pace_dec = (tempo_seg / 60.0) / (dist_trecho / 1000.0)
    vel_m_min = dist_trecho / (tempo_seg / 60.0)

    hr_seg = df_hr[(df_hr["ts_ms"] >= ts_ini) & (df_hr["ts_ms"] <= ts_fim)][
        "bpm"
    ]
    cad_seg = df_cad[
        (df_cad["ts_ms"] >= ts_ini) & (df_cad["ts_ms"] <= ts_fim)
    ]["cadencia"]
    alt_seg = df_alti[
        (df_alti["ts_ms"] >= ts_ini) & (df_alti["ts_ms"] <= ts_fim)
    ]["altitude_smooth"]

    bpm_min = hr_seg.min() if not hr_seg.empty else np.nan
    bpm_med = hr_seg.mean() if not hr_seg.empty else np.nan
    bpm_max = hr_seg.max() if not hr_seg.empty else np.nan

    cad_min = cad_seg.min() if not cad_seg.empty else np.nan
    cad_med = cad_seg.mean() if not cad_seg.empty else np.nan
    cad_max = cad_seg.max() if not cad_seg.empty else np.nan

    if len(alt_seg) >= 2:
      diffs_alt = alt_seg.diff()
      ganho_alt = diffs_alt[diffs_alt > 0].sum()
      perda_alt = diffs_alt[diffs_alt < 0].sum()
      saldo_alt = alt_seg.iloc[-1] - alt_seg.iloc[0]
      inclinacao_pct = (saldo_alt / dist_trecho) * 100.0
    else:
      ganho_alt, perda_alt, saldo_alt, inclinacao_pct = (
          np.nan,
          np.nan,
          np.nan,
          np.nan,
      )

    splits_km.append({
        "record_id": rec_id,
        "data": data_treino,
        "modalidade": modalidade,
        "km": km,
        "distancia_m": round(dist_trecho, 1),
        "tempo_seg": round(tempo_seg, 1),
        "duracao_min": round(tempo_seg / 60.0, 2),
        "duracao_fmt": formatar_duracao_segundos(tempo_seg),
        "pace_dec": round(pace_dec, 2),
        "pace_fmt": formatar_pace(pace_dec),
        "delta_pace_seg": round((pace_dec - pace_treino_dec) * 60.0, 1),
        "bpm_min": round(bpm_min, 1) if pd.notna(bpm_min) else np.nan,
        "bpm_medio": round(bpm_med, 1) if pd.notna(bpm_med) else np.nan,
        "bpm_max": round(bpm_max, 1) if pd.notna(bpm_max) else np.nan,
        "delta_bpm": (
            round(bpm_med - bpm_treino_med, 1)
            if pd.notna(bpm_med) and bpm_treino_med
            else np.nan
        ),
        "cadencia_min": round(cad_min, 1) if pd.notna(cad_min) else np.nan,
        "cadencia_media": round(cad_med, 1) if pd.notna(cad_med) else np.nan,
        "cadencia_max": round(cad_max, 1) if pd.notna(cad_max) else np.nan,
        "altitude_min_m": (
            round(alt_seg.min(), 1) if not alt_seg.empty else np.nan
        ),
        "altitude_media_m": (
            round(alt_seg.mean(), 1) if not alt_seg.empty else np.nan
        ),
        "altitude_max_m": (
            round(alt_seg.max(), 1) if not alt_seg.empty else np.nan
        ),
        "ganho_elev_m": round(ganho_alt, 1) if pd.notna(ganho_alt) else np.nan,
        "perda_elev_m": round(perda_alt, 1) if pd.notna(perda_alt) else np.nan,
        "saldo_elev_m": round(saldo_alt, 1) if pd.notna(saldo_alt) else np.nan,
        "inclinacao_pct": (
            round(inclinacao_pct, 2) if pd.notna(inclinacao_pct) else np.nan
        ),
        "eficiencia_vel_bpm": (
            round(vel_m_min / bpm_med, 3)
            if pd.notna(bpm_med) and bpm_med > 0
            else np.nan
        ),
        "origem_split": origem,
    })

  df_km_sessao = pd.DataFrame(splits_km)
  if df_km_sessao.empty:
    return None, None, None

  kms_cheios = df_km_sessao[df_km_sessao["distancia_m"] >= 800]
  if not kms_cheios.empty and (kms_cheios["pace_dec"] < 2.75).any():
    return None, None, None

  # --- 2. DF POR 100M ---
  splits_100m = []
  if len(rs_rows) >= 10:
    df_vel = pd.DataFrame(rs_rows, columns=["k_raw", "vel_ms"])
    if (df_vel["k_raw"] > 1e11).all():
      df_vel["ts_ms"] = df_vel["k_raw"].astype("int64")
      df_vel = (
          df_vel.sort_values("ts_ms")
          .drop_duplicates("ts_ms")
          .reset_index(drop=True)
      )
    else:
      df_vel = df_vel.sort_values("k_raw").reset_index(drop=True)
      df_vel["ts_ms"] = np.linspace(
          start_ms, end_ms, len(df_vel), dtype="int64"
      )

    slice_idx = 1
    for _, d_ini, d_fim, _, _, ts_ini, ts_fim, _ in janelas:
      sub_vel = df_vel[
          (df_vel["ts_ms"] >= ts_ini) & (df_vel["ts_ms"] <= ts_fim)
      ].copy()
      if len(sub_vel) < 2:
        continue

      if sub_vel["ts_ms"].iloc[0] > ts_ini:
        sub_vel = pd.concat(
            [
                pd.DataFrame(
                    {"ts_ms": [ts_ini], "vel_ms": [sub_vel["vel_ms"].iloc[0]]}
                ),
                sub_vel,
            ],
            ignore_index=True,
        )
      if sub_vel["ts_ms"].iloc[-1] < ts_fim:
        sub_vel = pd.concat(
            [
                sub_vel,
                pd.DataFrame(
                    {"ts_ms": [ts_fim], "vel_ms": [sub_vel["vel_ms"].iloc[-1]]}
                ),
            ],
            ignore_index=True,
        )

      dt = sub_vel["ts_ms"].diff().div(1000.0).fillna(0)
      dist_local = np.cumsum(sub_vel["vel_ms"] * dt).values
      if dist_local[-1] <= 0:
        continue

      dist_calib = d_ini + dist_local * ((d_fim - d_ini) / dist_local[-1])
      dist_calib = dist_calib + np.arange(len(dist_calib)) * 1e-5

      primeiro_marco = (int(d_ini // 100) + 1) * 100.0
      marcos_100 = list(np.arange(primeiro_marco, d_fim, 100.0))
      if not marcos_100 or (d_fim - marcos_100[-1]) >= 40.0:
        marcos_100.append(d_fim)
      else:
        marcos_100[-1] = d_fim

      ts_marcos = np.interp(marcos_100, dist_calib, sub_vel["ts_ms"].values)
      d_prev, ts_prev = d_ini, float(ts_ini)

      for d_curr, ts_curr in zip(marcos_100, ts_marcos):
        dist_m = d_curr - d_prev
        tempo_s = (ts_curr - ts_prev) / 1000.0

        if dist_m >= 40.0 and tempo_s > 1.0:
          pace_dec = (tempo_s / 60.0) / (dist_m / 1000.0)
          if 2.5 <= pace_dec <= 15.0:
            ts_mid = (ts_prev + ts_curr) / 2.0

            hr_seg = df_hr[
                (df_hr["ts_ms"] >= ts_prev) & (df_hr["ts_ms"] <= ts_curr)
            ]["bpm"]
            if not hr_seg.empty:
              b_min, b_med, b_max = hr_seg.min(), hr_seg.mean(), hr_seg.max()
            elif not df_hr.empty:
              b_med = float(np.interp(ts_mid, df_hr["ts_ms"], df_hr["bpm"]))
              b_min = b_max = b_med
            else:
              b_min = b_med = b_max = np.nan

            cad_seg = df_cad[
                (df_cad["ts_ms"] >= ts_prev) & (df_cad["ts_ms"] <= ts_curr)
            ]["cadencia"]
            if not cad_seg.empty:
              c_min, c_med, c_max = (
                  cad_seg.min(),
                  cad_seg.mean(),
                  cad_seg.max(),
              )
            elif not df_cad.empty:
              c_med = float(
                  np.interp(ts_mid, df_cad["ts_ms"], df_cad["cadencia"])
              )
              c_min = c_max = c_med
            else:
              c_min = c_med = c_max = np.nan

            if not df_alti.empty:
              alt_ini = float(
                  np.interp(
                      ts_prev, df_alti["ts_ms"], df_alti["altitude_smooth"]
                  )
              )
              alt_fim = float(
                  np.interp(
                      ts_curr, df_alti["ts_ms"], df_alti["altitude_smooth"]
                  )
              )
              saldo_elev = alt_fim - alt_ini
              inclinacao_pct = (saldo_elev / dist_m) * 100.0
            else:
              alt_ini = alt_fim = saldo_elev = inclinacao_pct = np.nan

            splits_100m.append({
                "record_id": rec_id,
                "data": data_treino,
                "modalidade": modalidade,
                "trecho_100m": slice_idx,
                "km_ref": int(np.ceil(d_curr / 1000.0)),
                "dist_acum_m": round(d_curr, 1),
                "distancia_m": round(dist_m, 1),
                "tempo_seg": round(tempo_s, 1),
                "duracao_min": round(tempo_s / 60.0, 2),
                "duracao_fmt": formatar_duracao_segundos(tempo_s),
                "pace_dec": round(pace_dec, 2),
                "pace_fmt": formatar_pace(pace_dec),
                "delta_pace_seg_km": round(
                    (pace_dec - pace_treino_dec) * 60.0, 1
                ),
                "bpm_min": round(b_min, 1) if pd.notna(b_min) else np.nan,
                "bpm_medio": round(b_med, 1) if pd.notna(b_med) else np.nan,
                "bpm_max": round(b_max, 1) if pd.notna(b_max) else np.nan,
                "delta_bpm": (
                    round(b_med - bpm_treino_med, 1)
                    if pd.notna(b_med) and bpm_treino_med
                    else np.nan
                ),
                "cadencia_min": round(c_min, 1) if pd.notna(c_min) else np.nan,
                "cadencia_media": (
                    round(c_med, 1) if pd.notna(c_med) else np.nan
                ),
                "cadencia_max": round(c_max, 1) if pd.notna(c_max) else np.nan,
                "altitude_ini_m": (
                    round(alt_ini, 1) if pd.notna(alt_ini) else np.nan
                ),
                "altitude_fim_m": (
                    round(alt_fim, 1) if pd.notna(alt_fim) else np.nan
                ),
                "saldo_elev_m": (
                    round(saldo_elev, 1) if pd.notna(saldo_elev) else np.nan
                ),
                "inclinacao_pct": (
                    round(inclinacao_pct, 2)
                    if pd.notna(inclinacao_pct)
                    else np.nan
                ),
            })
            slice_idx += 1

        d_prev, ts_prev = d_curr, ts_curr

  df_100m_sessao = pd.DataFrame(splits_100m)

  # --- 3. DF CORRIDAS (Resumo Macro) ---
  if len(df_hr) >= 10:
    meio_hr = len(df_hr) // 2
    deriva_hr = (
        df_hr["bpm"].iloc[meio_hr:].mean() / df_hr["bpm"].iloc[:meio_hr].mean()
    )
    pct_zona_alta = (df_hr["bpm"] >= 165).mean() * 100.0
  else:
    deriva_hr, pct_zona_alta = np.nan, np.nan

  bpm_min_s = s.get("minHeartRate")
  if not bpm_min_s or bpm_min_s <= 30:
    bpm_min_s = round(df_hr["bpm"].min(), 1) if not df_hr.empty else np.nan

  cad_min_s = round(df_cad["cadencia"].min(), 1) if not df_cad.empty else np.nan
  cad_max_s = s.get("bestStepRate")
  if not cad_max_s or cad_max_s <= 0:
    cad_max_s = (
        round(df_cad["cadencia"].max(), 1) if not df_cad.empty else np.nan
    )

  alt_min_s = s.get("mMinAlti", 0)
  if not alt_min_s or alt_min_s <= 0:
    alt_min_s = (
        round(df_alti["altitude"].min(), 1) if not df_alti.empty else np.nan
    )

  alt_max_s = s.get("mMaxAlti", 0)
  if not alt_max_s or alt_max_s <= 0:
    alt_max_s = (
        round(df_alti["altitude"].max(), 1) if not df_alti.empty else np.nan
    )

  wear = s.get("wearSportData") or {}

  resumo_treino = {
      "record_id": rec_id,
      "data": data_treino,
      "modalidade": modalidade,
      "distancia_km": round(dist_km, 2),
      "duracao_min": round(duracao_min, 2),
      "duracao_fmt": formatar_duracao_segundos(total_time_s),
      "pace_dec": round(pace_treino_dec, 2),
      "pace_fmt": formatar_pace(pace_treino_dec),
      "bpm_min": bpm_min_s,
      "bpm_medio": bpm_treino_med,
      "bpm_max": s.get("maxHeartRate"),
      "deriva_cardiaca": (
          round(deriva_hr, 3) if not np.isnan(deriva_hr) else np.nan
      ),
      "pct_tempo_bpm_165+": (
          round(pct_zona_alta, 1) if not np.isnan(pct_zona_alta) else np.nan
      ),
      "cadencia_min": cad_min_s,
      "cadencia_media": s.get("avgStepRate"),
      "cadencia_max": cad_max_s,
      "altitude_min_m": alt_min_s,
      "altitude_media_m": (
          round(df_alti["altitude"].mean(), 1) if not df_alti.empty else np.nan
      ),
      "altitude_max_m": alt_max_s,
      "ganho_elev_total_m": round(df_km_sessao["ganho_elev_m"].sum(), 1),
      "perda_elev_total_m": round(df_km_sessao["perda_elev_m"].sum(), 1),
      "kcal": round(s.get("totalCalories", 0) / 1000.0, 1),
      "te_aerobico": wear.get("etraining_effect", 0) / 10.0,
      "te_anaerobico": wear.get("anaerobic_exercise_etraining_effect", 0) / 10.0,
      "tem_splits_oficiais": janelas[0][7] == "oficial_relogio",
  }

  return resumo_treino, df_km_sessao, df_100m_sessao


# === EXECUÇÃO DO PIPELINE ===
todos_treinos, pasta_motion = localizar_treinos()
resumos, lista_km, lista_100m = [], [], []

for t in todos_treinos:
  r, d_km, d_100m = processar_tudo(t)
  if r is not None:
    resumos.append(r)
    lista_km.append(d_km)
    if d_100m is not None and not d_100m.empty:
      lista_100m.append(d_100m)

df_corridas = pd.DataFrame(resumos).sort_values("data").reset_index(drop=True)
df_por_km = (
    pd.concat(lista_km, ignore_index=True)
    .sort_values(["data", "km"])
    .reset_index(drop=True)
)
df_por_100m = (
    pd.concat(lista_100m, ignore_index=True)
    .sort_values(["data", "trecho_100m"])
    .reset_index(drop=True)
)

# --- Enriquecimento temporal ---
df_corridas["ano"] = df_corridas["data"].dt.year
df_corridas["mes"] = df_corridas["data"].dt.strftime("%Y-%m")
df_corridas["dia_semana"] = df_corridas["data"].dt.day_name()
df_corridas["hora"] = df_corridas["data"].dt.hour
df_corridas["horario"] = df_corridas["data"].dt.strftime("%H:%M")
df_corridas["periodo_dia"] = pd.cut(
    df_corridas["hora"],
    bins=[-1, 4, 11, 17, 23],
    labels=["Madrugada", "Manhã", "Tarde", "Noite"],
)

# --- Faixas de distância ---
ordem_faixas = [
    "< 2K",
    "Curto (2-4.89K)",
    "Base 5K",
    "5-6K",
    "Base 6K",
    "6-7K",
    "Base 7K",
    "7-8K",
    "Base 8K",
    "8-10K",
    "Base 10K",
    "+10K",
]
df_corridas["faixa_distancia"] = pd.cut(
    df_corridas["distancia_km"],
    bins=[0.0, 1.99, 4.89, 5.4, 5.9, 6.3, 6.9, 7.3, 7.9, 8.3, 9.8, 10.3, 50],
    labels=ordem_faixas,
    include_lowest=True,
)
df_corridas["faixa_distancia"] = pd.Categorical(
    df_corridas["faixa_distancia"], categories=ordem_faixas, ordered=True
)
bases_fechadas = ["Base 5K", "Base 6K", "Base 7K", "Base 8K", "Base 10K"]
df_corridas["distancia_fechada"] = df_corridas["faixa_distancia"].isin(
    bases_fechadas
)

# --- Peso Corporal (interpolação linear pelo tempo, sem NaNs) ---
df_peso = extrair_historico_peso(pasta_motion.parent)
if not df_peso.empty:
  ts_peso = (
      pd.to_datetime(df_peso["data_peso"])
      .astype("datetime64[s]")
      .astype("int64")
      .to_numpy()
  )
  vals_peso = df_peso["peso_kg"].to_numpy()
  ts_corridas = (
      pd.to_datetime(df_corridas["data"])
      .astype("datetime64[s]")
      .astype("int64")
      .to_numpy()
  )
  df_corridas["peso_kg"] = np.round(
      np.interp(ts_corridas, ts_peso, vals_peso), 2
  )
else:
  df_corridas["peso_kg"] = np.nan

# Propaga peso_kg, faixa_distancia e distancia_fechada para 1km e 100m
mapa_peso = df_corridas.set_index("record_id")["peso_kg"]
mapa_faixa = df_corridas.set_index("record_id")["faixa_distancia"]
mapa_fechada = df_corridas.set_index("record_id")["distancia_fechada"]

for df_detalhe in (df_por_km, df_por_100m):
  df_detalhe["peso_kg"] = df_detalhe["record_id"].map(mapa_peso)
  df_detalhe["faixa_distancia"] = df_detalhe["record_id"].map(mapa_faixa)
  df_detalhe["distancia_fechada"] = df_detalhe["record_id"].map(mapa_fechada)

# Defasagens de inclinação (100m e 200m antes) para inércia cardíaca
df_por_100m["inclinacao_lag1_pct"] = df_por_100m.groupby("record_id")[
    "inclinacao_pct"
].shift(1)
df_por_100m["inclinacao_lag2_pct"] = df_por_100m.groupby("record_id")[
    "inclinacao_pct"
].shift(2)

# Auditoria rápida para flagrar futuros treinos gravados errado
suspeitos = df_corridas[
    (df_corridas["modalidade"] == "Rua")
    & (
        (df_corridas["hora"] < 12)
        | (
            (df_corridas["pace_dec"] < 8.0)
            & (df_corridas["cadencia_media"].fillna(0) < 140)
        )
    )
][["record_id", "data", "distancia_km", "pace_fmt", "bpm_medio", "cadencia_media"]]

if not suspeitos.empty:
  print("\n=== TREINOS SUSPEITOS ===")
  print(suspeitos.to_string(index=False))

# Salva os 3 CSVs consolidados
df_corridas.to_csv("df_corridas.csv", index=False)
df_por_km.to_csv("df_por_km.csv", index=False)
df_por_100m.to_csv("df_por_100m.csv", index=False)

print("\nBases consolidadas e salvas:")
print(f" - df_corridas.csv : {df_corridas.shape} | peso preenchido: {df_corridas['peso_kg'].notna().sum()}/{len(df_corridas)}")
print(f" - df_por_km.csv   : {df_por_km.shape}")
print(f" - df_por_100m.csv : {df_por_100m.shape}")