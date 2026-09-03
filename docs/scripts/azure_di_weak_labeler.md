# Azure DI Weak Labeler (`azure_di_weak_labeler.py`)

## Overview

`azure_di_weak_labeler.py` è uno script da riga di comando che genera annotazioni "deboli" in formato COCO per una directory di immagini utilizzando **Azure Document Intelligence** (modello `prebuilt-layout`).

A differenza di `local_weak_labeler.py` — che carica un modello RF-DETR localmente — questo script invia ogni immagine all'API cloud di Azure e riceve in risposta le posizioni e lo stato delle **checkbox** (spunte/caselle di selezione) rilevate nei documenti. È progettato per il **weak labeling automatico** di dataset di immagini contenenti moduli con checkbox, producendo un file JSON nel medesimo formato COCO degli altri labeler del progetto.

---

## Prerequisiti

1. **Credenziali Azure DI:** Un endpoint e una API key di Azure Document Intelligence devono essere disponibili e configurati nel file `.env.local` (vedi [Configurazione](#configurazione)).
2. **File `.env.local`:** Deve esistere alla radice del progetto. Il file è escluso da git tramite `.gitignore` e non deve mai essere committato.
3. **Dipendenze:** Richiede `azure-ai-documentintelligence`, `python-dotenv`, `pydantic-settings`, `pillow` e `tqdm`. Tutte sono elencate in `pyproject.toml`.
4. **Formato immagini:** Supporta file `.jpg`, `.jpeg` e `.png` (case-insensitive).
5. **Connessione di rete:** Richiede accesso all'endpoint Azure DI configurato.

---

## Configurazione

La configurazione avviene esclusivamente tramite **variabili d'ambiente**, caricate automaticamente dal file `.env.local` alla radice del progetto.

### File `.env.local`

```dotenv
# .env.local  —  NON committare questo file

# Endpoint primario Azure Document Intelligence
AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/

# API Key primaria
AZURE_DI_KEY=<your-api-key>

# (Opzionale) Endpoint e key di disaster recovery
AZURE_DI_ENDPOINT_DR=https://<your-dr-resource>.cognitiveservices.azure.com/
AZURE_DI_KEY_DR=<your-dr-api-key>
```

> **Sicurezza:** Il file `.env.local` è incluso in `.gitignore` e non viene mai tracciato da git. Non inserire le credenziali direttamente nel codice o negli argomenti CLI.

### Variabili d'ambiente disponibili

| Variabile | Obbligatoria | Default | Descrizione |
|-----------|:---:|---------|-------------|
| `AZURE_DI_ENDPOINT` | Sì | — | URL dell'endpoint Azure Document Intelligence |
| `AZURE_DI_KEY` | Sì | — | API key per l'autenticazione |
| `AZURE_DI_MODEL_ID` | No | `prebuilt-layout` | Modello Azure DI da utilizzare |
| `AZURE_DI_CONFIDENCE_THRESHOLD` | No | `0.80` | Soglia minima di confidenza per includere una detection |
| `AZURE_DI_TIMEOUT` | No | `600` | Timeout in secondi per ogni chiamata API |

> **Importante:** Il modello `prebuilt-read` (default del servizio Azure) **non rileva checkbox**. Usare sempre `prebuilt-layout` o un modello personalizzato che supporti i `selectionMarks`.

### Priorità di risoluzione della configurazione

| Priorità | Sorgente | Esempio |
|----------|----------|---------|
| 1 (massima) | Argomenti CLI | `--confidence 0.90`, `--model prebuilt-layout` |
| 2 | Variabili d'ambiente (`AZURE_DI_*`) | `AZURE_DI_CONFIDENCE_THRESHOLD=0.85` |
| 3 (minima) | Default Python | `confidence_threshold = 0.80` |

---

## Come Funziona

1. **Caricamento configurazione:** Le impostazioni vengono caricate da `.env.local` (e/o variabili d'ambiente di sistema) tramite `AzureSettings` in `src/azure_labeling/azure_config.py`.
2. **Costruzione client:** Viene istanziato un `DocumentIntelligenceClient` autenticato con endpoint e API key. In modalità dry-run il client non viene mai creato.
3. **Scansione:** Tutti i file immagine validi (JPG e PNG) nella directory di input vengono raccolti, con scansione opzionale delle sottodirectory tramite `--recursive`.
4. **Controllo resume/overwrite:** Se `--resume` è attivo e il file di output esiste già, lo script carica le annotazioni esistenti e salta le immagini già elaborate. Se `--overwrite` è attivo, procede senza chiedere conferma.
5. **Loop di inferenza:** Per ogni nuova immagine, lo script:
   - Legge le dimensioni dell'immagine con PIL (`width`, `height`).
   - Invia i byte dell'immagine ad Azure DI con `begin_analyze_document(model_id, ...)`.
   - Attende il completamento asincrono (pattern `POST 202 → polling GET → risposta finale`).
   - Estrae i `selection_marks` dalla risposta per ogni pagina del documento.
   - Filtra le detection con `confidence >= confidence_threshold`.
6. **Conversione bbox:** Azure DI restituisce i poligoni con coordinate **normalizzate** (0.0–1.0). Lo script le converte in pixel assoluti moltiplicando per `width`/`height`, poi produce la bbox COCO `[x, y, w, h]`:
   ```
   x_abs = x_norm * width
   y_abs = y_norm * height
   x = min(x_abs), y = min(y_abs)
   w = max(x_abs) - min(x_abs)
   h = max(y_abs) - min(y_abs)
   ```
7. **Classificazione:** Lo stato della checkbox viene mappato nelle categorie COCO:
   - `selected` → `category_id: 1`, label `"checked"`
   - `unselected` → `category_id: 2`, label `"unchecked"`
8. **Export:** Il file COCO JSON viene scritto nel percorso di output specificato.

---

## Formato di Output

Il file di output è un COCO JSON standard con la seguente struttura:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "modulo_001.jpg",
            "width": 2480,
            "height": 3508
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [312, 540, 48, 47],
            "area": 2256,
            "iscrowd": 0,
            "score": 0.9812
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": [312, 640, 48, 47],
            "area": 2256,
            "iscrowd": 0,
            "score": 0.9543
        }
    ],
    "categories": [
        {"id": 1, "name": "checked",   "supercategory": "checkbox"},
        {"id": 2, "name": "unchecked", "supercategory": "checkbox"}
    ]
}
```

Il campo `score` corrisponde alla `confidence` restituita da Azure DI per ogni `selectionMark`.

---

## Utilizzo

Eseguire lo script dalla radice del progetto:

```bash
python scripts/azure_di_weak_labeler.py -i "path/to/images" -o "path/to/output.json"
```

### Argomenti

| Argomento | Flag corto | Obbligatorio | Default | Descrizione |
|-----------|-----------|:---:|---------|-------------|
| `--input` | `-i` | Sì | — | Directory contenente le immagini da etichettare |
| `--output` | `-o` | Sì | — | Percorso del file COCO JSON di output |
| `--confidence` | — | No | *(da config)* | Override della soglia di confidenza (0.0–1.0) |
| `--model` | — | No | `prebuilt-layout` | Override del modello Azure DI |
| `--recursive` | `-r` | No | `False` | Scansiona le sottodirectory ricorsivamente |
| `--resume` | — | No | `False` | Riprende da un file di output esistente, saltando le immagini già elaborate |
| `--overwrite` | — | No | `False` | Sovrascrive il file di output senza chiedere conferma |
| `--dry-run` | — | No | `False` | Simula l'esecuzione: scansiona e conta le immagini senza chiamare Azure né scrivere file |

> **Nota:** `--resume` e `--overwrite` sono mutualmente esclusivi. Usarli insieme produce un errore.

### Esempi

**Utilizzo base:**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json"
```

**Scansione ricorsiva delle sottodirectory:**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/" \
    -o "data/datasets/weak_labels_azure_all.json" \
    --recursive
```

**Dry run (anteprima senza chiamate Azure):**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json" \
    --dry-run
```

**Override della soglia di confidenza:**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json" \
    --confidence 0.90
```

**Riprendere un job interrotto:**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json" \
    --resume
```

**Sovrascrittura silenziosa (utile in pipeline CI):**
```bash
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json" \
    --overwrite
```

**Override del modello via variabile d'ambiente:**
```bash
AZURE_DI_MODEL_ID="prebuilt-document" \
python scripts/azure_di_weak_labeler.py \
    -i "data/raw/batch_3" \
    -o "data/datasets/weak_labels_azure_batch_3.json"
```

---

## Modalità Dry Run

Quando viene passato `--dry-run`, lo script scansiona le immagini e riporta cosa accadrebbe, senza instanziare il client Azure, effettuare chiamate API o scrivere file.

Output di esempio:
```
[DRY RUN] Input directory  : data/raw/batch_3
[DRY RUN] Output file      : data/datasets/weak_labels_azure_batch_3.json
[DRY RUN] Azure DI model   : prebuilt-layout
[DRY RUN] Confidence thresh: 0.8
[DRY RUN] Images found     : 10
[DRY RUN] Would process    : 10
[DRY RUN] No inference was run. No files were written.
```

---

## Resume / Checkpoint

Se un job viene interrotto (es. timeout di rete, shutdown), può essere ripreso senza rielaborare le immagini già completate.

```bash
python scripts/azure_di_weak_labeler.py -i "data/raw/batch_3" -o "out.json" --resume
```

**Come funziona:**
- Se il file di output esiste: le immagini e le annotazioni esistenti vengono caricate; vengono elaborate solo le immagini il cui `file_name` non è già presente nell'output; i contatori di `image_id` e `annotation_id` ripartono dall'ultimo valore salvato.
- Se il file di output non esiste: lo script parte da zero senza errori.
- Al termine, il file di output contiene sia le annotazioni precedenti sia quelle nuove.

---

## Protezione del File di Output

| Flag utilizzati | File di output esistente? | Comportamento |
|----------------|:---:|---------------|
| *(nessuno)* | No | Parte normalmente |
| *(nessuno)* | Sì | Chiede conferma interattiva: `"Output file already exists. Overwrite? [y/N]"` |
| `--overwrite` | No | Parte normalmente |
| `--overwrite` | Sì | Sovrascrive senza prompt |
| `--resume` | No | Parte da zero (nessun errore) |
| `--resume` | Sì | Carica i dati esistenti, aggiunge le nuove annotazioni |
| `--resume --overwrite` | qualsiasi | **Errore** — flag mutuamente esclusivi |

---

## Gestione degli Errori

Se un'immagine non può essere letta o la chiamata Azure fallisce, lo script cattura l'eccezione, registra l'errore tramite `tqdm.write()` (senza interrompere la progress bar) e procede con l'immagine successiva. Un singolo fallimento non interrompe l'intero job.

---

## Struttura del Modulo

```
src/azure_labeling/
    __init__.py             # Package init — esporta generate_azure_weak_labels, AzureSettings
    azure_config.py         # AzureSettings (pydantic-settings + .env.local), categorie COCO
    azure_di_labeler.py     # Logica core: client Azure, analisi immagini, conversione bbox

scripts/
    azure_di_weak_labeler.py    # Entry point CLI

.env.local                      # Credenziali Azure DI (escluso da git)
```