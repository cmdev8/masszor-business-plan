# Masszőr bevétel előrejelzés – Monte Carlo

Valószínűség-alapú (Monte Carlo) havi bevétel-előrejelzés 1 személyes masszőr vállalkozáshoz.

## Funkciók
- Havi bontás (12 hónap)
- Szezonális új ügyfelek (hónaponként min/mode/max)
- Lemorzsolódás (churn)
- Ismétlődő alkalmak (keverék)
- Kapacitáskorlát (munkanapok, napi órák, kezelés + csere idő)
- Lemondás/no-show
- Marketing költség: fix/hó + CAC (Ft/új ügyfél)
- Export: CSV letöltés

## Lokális futtatás
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deploy
1. Tedd fel ezt a repót GitHubra (public).
2. Streamlit Cloud: New app → válaszd a repót → Main file: `app.py` → Deploy.
