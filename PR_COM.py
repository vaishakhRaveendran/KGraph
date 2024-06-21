import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv
from groq import Groq
import scipy as sp

load_dotenv()  # Load environment variables from a .env file

class DocumentGraphQuestionGenerator:
    def __init__(self, model="llama3-8b-8192", similarity_threshold=0.5):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("API key not found. Please set the GROQ_API_KEY environment variable.")
            return
        self.client = Groq(
            api_key=api_key,
        )
        self.similarity_threshold = similarity_threshold
        self.model = model

    def get_embedding(self, text):
        self.embed =  OpenAI()
        response = self.embed.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding 


    def create_sparse_matrix(self, data, row, col, shape, dtype):
        return sp.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=dtype)

    def create_semantic_graph(self, chunks):
        embeddings = [self.get_embedding(chunk) for chunk in chunks]
        similarity_matrix = cosine_similarity(embeddings)

        nlen = len(chunks)
        rows, cols = np.where(similarity_matrix > self.similarity_threshold)
        data = similarity_matrix[rows, cols]

        sparse_matrix = self.create_sparse_matrix(data, rows, cols, (nlen, nlen), dtype=float)
        G = nx.from_scipy_sparse_array(sparse_matrix)

        for i, chunk in enumerate(chunks):
            G.nodes[i]['content'] = chunk

        return G

    def detect_communities(self, G):
        return list(nx.community.louvain_communities(G))

    def rank_nodes_in_community(self, G, community):
        subgraph = G.subgraph(community)
        return nx.pagerank(subgraph)

    def generate_question(self, context):
        prompt = f"Generate a thought-provoking question based on the following context:\n\n{context}\n\nQuestion:"
        self.chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        return self.chat_completion.choices[0].message.content
    
    def generate_questions(self, G, communities, num_questions_per_community=2):
        questions = []

        for community in communities:
            subgraph = G.subgraph(community)
            pagerank = self.rank_nodes_in_community(G, community)
            top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:3]

            context = " ".join([G.nodes[node]['content'] for node in top_nodes])

            for _ in range(num_questions_per_community):
                question = self.generate_question(context)
                questions.append(question)

        return questions

    def generate_inter_community_questions(self, G, communities, num_questions=3):
        inter_community_questions = []

        for _ in range(num_questions):
            # Randomly select two different communities
            comm1, comm2 = np.random.choice(len(communities), 2, replace=False)

            # Select a random node from each community
            node1 = np.random.choice(list(communities[comm1]))
            node2 = np.random.choice(list(communities[comm2]))

            # Combine the content of these nodes
            context = G.nodes[node1]['content'] + " " + G.nodes[node2]['content']

            question = self.generate_question(context)
            inter_community_questions.append(question)

        return inter_community_questions

    def process_document(self, chunks, num_intra_questions_per_community=2, num_inter_community_questions=3):
        G = self.create_semantic_graph(chunks)
        communities = self.detect_communities(G)

        intra_community_questions = self.generate_questions(G, communities, num_intra_questions_per_community)
        inter_community_questions = self.generate_inter_community_questions(G, communities, num_inter_community_questions)

        return intra_community_questions, inter_community_questions

# Example usage
if __name__ == "__main__":
    chunks = [
        "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water within the Earth and atmosphere.",
        "It is a complex system that includes many different processes: evaporation, transpiration, condensation, precipitation, and runoff.",
        "Climate change is a long-term change in the average weather patterns that have come to define Earth's local, regional, and global climates.",
        "The primary cause of climate change is human activities, particularly the burning of fossil fuels, which adds heat-trapping greenhouse gases to Earth's atmosphere.",
        "Renewable energy is energy derived from natural sources that are replenished at a higher rate than they are consumed.",
        "Sunlight, wind, rain, tides, waves, and geothermal heat are all renewable resources that can be used to produce sustainable energy."
    ]

    generator = DocumentGraphQuestionGenerator()
    intra_questions, inter_questions = generator.process_document(chunks)

    print("Intra-community questions:")
    for q in intra_questions:
        print(f"- {q}")

    print("\nInter-community questions:")
    for q in inter_questions:
        print(f"- {q}") 
