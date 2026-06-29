from src.ingestion.nse.nse_corp_actions import ingest_once

if __name__ == "__main__":
    fetched, inserted = ingest_once()
    print(f"Fetched={fetched}, Inserted={inserted}")