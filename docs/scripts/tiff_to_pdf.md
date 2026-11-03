# Conversione TIFF in PDF (`tiff_to_pdf.py`)

## Panoramica

Lo script `scripts/tiff_to_pdf.py` converte un'immagine TIFF in un file PDF. Supporta TIFF a pagina singola e multipagina; in quest'ultimo caso tutte le pagine vengono incluse nel PDF.

La logica di conversione è implementata in `src/utils/tiff_to_pdf.py`. Lo script può essere eseguito dalla cartella principale del repository.

## Prerequisiti

- Python e le dipendenze del progetto installate nell'ambiente virtuale. La conversione usa Pillow, già elencato in `pyproject.toml`.
- Un file TIFF esistente e leggibile.

## Utilizzo

Attivare l'ambiente virtuale, aprire un terminale nella cartella principale del repository e passare il file TIFF come argomento:

```powershell
python scripts/tiff_to_pdf.py "data/documento.tiff"
```

Se non si specifica un percorso di output, lo script salva il PDF nella stessa cartella dell'immagine, mantenendo il nome e sostituendo l'estensione con `.pdf`. Nell'esempio, il risultato sarà `data/documento.pdf`.

Per scegliere un percorso di output diverso, usare `-o` oppure `--output`:

```powershell
python scripts/tiff_to_pdf.py "data/documento.tiff" --output "output/documento_convertito.pdf"
```

Le cartelle di destinazione mancanti vengono create automaticamente.

## Argomenti

| Argomento | Obbligatorio | Descrizione |
|---|:---:|---|
| `input` | Sì | Percorso del file TIFF da convertire. |
| `-o`, `--output` | No | Percorso del PDF da creare. Se omesso, viene usato il percorso del file input con estensione `.pdf`. |
| `-h`, `--help` | No | Mostra l'aiuto della riga di comando. |

Per visualizzare l'aiuto:

```powershell
python scripts/tiff_to_pdf.py --help
```

## Comportamento ed errori

- Il file d'ingresso deve essere un TIFF riconosciuto da Pillow.
- Il percorso di output deve avere estensione `.pdf`.
- Input e output non possono indicare lo stesso file.
- Se il percorso di output esiste già, il file viene sovrascritto.
- In caso di errore, il messaggio viene riportato nel terminale e lo script termina con codice `1`; in caso di successo termina con codice `0`.