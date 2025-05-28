
from flask import Flask, render_template, request, jsonify
from iris_db import  ask_query_rag,ask_query_graphrag,ask_query_graphrag_with_docs,ask_query_no_rag,load_graph_data
# from sentence_transformers import SentenceTransformer
import pandas as pd
from flask import request, session, jsonify  # make sure you imported these at top
import hashlib
from iris_db import setup_environment

setup_environment()


graph_cache = {}


papers_df = pd.read_csv('CSV/papers300.csv')
entities_df = pd.read_csv('CSV/entities300.csv')
relations_df = pd.read_csv('CSV/relations300.csv')


app = Flask(__name__)
# app.secret_key = ""  # Required for session to work


@app.route("/", methods=["GET", "POST"])
def home():
    question = ""  # Default empty question
    answer1 = None
    answer2 = None

    if request.method == "POST":
        question = request.form.get("question")
        action = request.form.get("action")  # Which button was clicked

        answer1 = ask_query_rag(question, graphitems=0, vectoritems=50)
        answer2 = ask_query_graphrag(question, graphitems=50, vectoritems=0)
     
        

    return render_template("index.html", question=question, answer1=answer1, answer2=answer2)




@app.route("/api/graph")
def get_graph():
    nodes = {}
    links = []

# Extract paper titles from rows where type == 'Paper'
    paper_titles = dict(zip(papers_df['docid'], papers_df['title']))


    # Paper nodes from entities table
    # for doc_id in entities_df['docid'].unique():
    #     paper_id = f"Paper_{doc_id}"
    #     title = paper_titles.get(doc_id, f"(Untitled Paper {doc_id})")

    #     nodes[paper_id] = {
    #         "id": paper_id,
    #         "type": "Paper",
    #         "label": title,
    #     }



    # Entity nodes + links
    for _, row in entities_df.iterrows():
        docid = row['docid']
        paper_id = f"Paper_{docid}"
        node_type = row['type']

        if node_type == "Paper":
            # Don't add a duplicate paper node
            # Just link to the paper_id correctly if needed
            continue

        entity_id = row['entityid'].strip().lower()

        if entity_id not in nodes:
            nodes[entity_id] = {
                "id": entity_id,
                "type": node_type,
                "label": row['entityid']  # Keep original case
            }

        links.append({
            "source": entity_id,
            "target": paper_id,
            "type": "AUTHORED" if node_type == "Author" else "COVERS"
        })


    # Relation nodes + links
# Only create paper nodes from the papers CSV
    for _, row in papers_df.iterrows():
        paper_id = f"Paper_{row['docid']}"
        title = row['title']
        nodes[paper_id] = {
            "id": paper_id,
            "type": "Paper",
            "label": title[:100],  # Optional: truncate very long titles
            "url": row.get('url', '')
        }


    return jsonify({"nodes": list(nodes.values()), "links": links})

@app.route("/api/store_graph_docs", methods=["POST"])
def store_graph_docs():
    query = request.json.get("query")
    app.logger.info(f"📥 Received query for graph: {query}")

    raw_text = ask_query_graphrag_with_docs(query)
    app.logger.info(f"📦 Retrieved {len(raw_text)} docs. Sample:\n{raw_text[0][:500] if raw_text else 'No data'}")

    # Create a lightweight key for the session
    query_key = hashlib.md5(query.encode()).hexdigest()
    graph_cache[query_key] = raw_text

    session["graph_key"] = query_key  # Now session stores only a tiny string

    return jsonify({"status": "ok"})



@app.route("/mode1", methods=["GET", "POST"])
def mode1():
    question = ""
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        answer = ask_query_no_rag(question)
    return render_template("mode1.html", question=question, answer=answer, current_mode="mode1")

@app.route("/mode2", methods=["GET", "POST"])
def mode2():
    question = ""
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        answer = ask_query_rag(question, graphitems=0, vectoritems=50)
    return render_template("mode2.html", question=question, answer=answer, current_mode="mode2")

@app.route("/mode3", methods=["GET", "POST"])
def mode3():
    question = ""
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        answer = ask_query_rag(question, graphitems=50, vectoritems=0)
    return render_template("mode3.html", question=question, answer=answer, current_mode="mode3")


@app.route("/mode4", methods=["GET", "POST"])
def mode4():
    question = ""  # Default empty question
    answer1 = None
    answer2 = None

    if request.method == "POST":
        question = request.form.get("question")
        # action = request.form.get("action")  # Which button was clicked

       
        answer1 = ask_query_no_rag(question)
     
        answer2 = ask_query_rag(question, graphitems=0, vectoritems=50)

    return render_template("mode4.html", question=question, answer1=answer1, answer2=answer2, current_mode="mode4")

@app.route("/mode5", methods=["GET", "POST"])
def mode5():
    question = ""  # Default empty question
    answer1 = None
    answer2 = None

    if request.method == "POST":
        question = request.form.get("question")
        action = request.form.get("action")  # Which button was clicked

         
        answer1 = ask_query_rag(question, graphitems=0, vectoritems=50)
        answer2 = ask_query_graphrag(question, graphitems=50, vectoritems=0)
   

    return render_template("mode5.html", question=question, answer1=answer1, answer2=answer2, current_mode="mode5")


@app.route("/api/query-graph")
def query_graph():
    query_key = session.get("graph_key")
    if not query_key or query_key not in graph_cache:
        return jsonify({"nodes": [], "links": []})

    context = graph_cache[query_key]
    nodes = {}
    links = {}

    # Step 1: Extract relevant docids from cached context
    matching_titles = []
    for abstract in context:
        title_line = next((line for line in abstract.split("\n") if "TITLE:" in line.upper()), None)
        if title_line:
            title = title_line.split("TITLE:", 1)[-1].strip()
            matching_titles.append(title.lower())

    # Step 2: Map titles to docids
    title_to_docid = {title.lower(): docid for docid, title in zip(papers_df['docid'], papers_df['title'])}
    matched_docids = [docid for title, docid in title_to_docid.items() if title in matching_titles]

    # Step 3: Build graph only for matched docids using entities_df
    for _, row in entities_df[entities_df['docid'].isin(matched_docids)].iterrows():
        docid = row['docid']
        paper_id = f"Paper_{docid}"
        node_type = row['type']
        entity_id = row['entityid'].strip()

        # Add paper node (only once)
        if paper_id not in nodes:
            paper_title = papers_df.loc[papers_df['docid'] == docid, 'title'].values[0]
            nodes[paper_id] = {
                "id": paper_id,
                "type": "Paper",
                "label": paper_title
            }

        if node_type != "Paper":
            entity_node_id = f"{node_type}_{entity_id.lower().replace(' ', '_')}"
            if entity_node_id not in nodes:
                nodes[entity_node_id] = {
                    "id": entity_node_id,
                    "type": node_type,
                    "label": entity_id
                }

            links[(entity_node_id, paper_id)] = {
                "source": entity_node_id,
                "target": paper_id,
                "type": "AUTHORED" if node_type == "Author" else "COVERS"
            }

    return jsonify({
        "nodes": list(nodes.values()),
        "links": list(links.values())
    })

if __name__ == "__main__":
    load_graph_data()

    app.run(host="0.0.0.0", port=5000, debug=True)







