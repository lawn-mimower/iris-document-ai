import argparse

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Default queries (form fields to look up)
queries = [
    #Page 1 questions
            "Language (English/Hindi)",
            "Authorised capital of the company as on the date of filling (in Rs.)",
            "1.(a) Corporate Identity Number (CIN) of company",
            "2.(a) Name of the company",
            "2.(b) Address of the registered office of the company",
            "2.(c) e-mail ID of the company",
            "3. Financial year to which financial statement relates - From:____ (DD/MM/YYYY) To:____ (DD/MM/YYYY)",
            "4.(a) Date of Board of Director's meeting in which financial statements are approved - ____ (DD/MM/YYYY)",
            "4.(b) Nature of financial statements: ____? 4.(c) Whether provisional financial statements filed earlier (Yes/No/Not appicable). 4.(d) Whether adopted in adjourned AGM (Yes/No/Not appicable)",
            "5.(a) Whether Annual General Meeting (AGM) held (Yes/No/Not appicable), 5.(b) if YES then date of AGM:____ (DD/MM/YYYY). 5.(c) Due date of AGM:____ (DD/MM/YYYY). 5.(d) Whether any extension for financial year or AGM granted (Yes/No)",
            "6.(a) Whether Schedule 3 of the Companies Act, 2013 is applicable (Yes/No). 6.(b) Whether financial statements have been drawn on the basis of ____ (AS Taxonomy/Ind-AS Taxonomy)",

        #Page 2 questions
            "7. Type of Industry",
            "8. Whether consolidated Financial State,emts are also being filed (Yes/No)",
            "9.(a) In case if a government company, whether CAG of India has commented upon of supplemented the audit report under section 143 of the Companies Act, 2013 (Yes/No)",
            "9.(d) whether CAG of India has conducted supplementary or test audit under section 143 (Yes/No)",
            "10. Whether Secretarial Audit is applicable (Yes/No)",
            "11. Whether detailed discloure with respect to Director's report Sec 134(3) is attached (Yes/No)"
    ]


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks with their source location from a Milvus Lite collection.")
    parser.add_argument("--db", default="oneshot.db", help="Milvus Lite database file (default: oneshot.db)")
    parser.add_argument("--collection", default="oneshot", help="Collection name (default: oneshot)")
    parser.add_argument("--query", "-q", action="append", help="Query text (repeatable; default: built-in form field queries)")
    parser.add_argument("--limit", type=int, default=5, help="Results per query (default: 5)")
    args = parser.parse_args()

    # Load model and connect
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    client = MilvusClient(args.db)

    for query in args.query or queries:
        query_embedding = model.encode([query])[0].tolist()
        results = client.search(
            collection_name=args.collection,
            data=[query_embedding],
            limit=args.limit,
            output_fields=["text", "filename", "page_number", "bbox_l", "bbox_t", "bbox_r", "bbox_b", "item_type"]
        )

        print(f"\nQuery: {query}")
        # Results contain full visual grounding metadata
        # Use bbox coordinates to highlight source on original document!
        for result in results[0]:
            print(f"File: {result['entity']['filename']}")
            print(f"Page: {result['entity']['page_number']}")
            print(f"BBox: ({result['entity']['bbox_l']}, {result['entity']['bbox_t']}, "
                f"{result['entity']['bbox_r']}, {result['entity']['bbox_b']})")
            print(f"Text: {result['entity']['text']}")


if __name__ == "__main__":
    main()