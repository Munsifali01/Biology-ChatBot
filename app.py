# app.py — Single-file MDCAT Biology Chatbot (11th + 12th)
# How to run:
#   pip install streamlit scikit-learn pandas numpy
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MDCAT Biology Helper Bot", page_icon="🧬")

st.title("MDCAT Biology Helper Bot")
st.caption("11th & 12th (F.Sc) + MDCAT-style Biology Q/A. Type a question and get an instant answer. Upload more MCQs if you want—no extra files required.")

# -----------------------------
# Built-in starter dataset (100 Q/A)
# -----------------------------
MCQS = [
    # ----- Class 11: Cell Structure -----
    {"class":"11","chapter":"Cell Structure","question":"What is the basic unit of life?","answer":"Cell"},
    {"class":"11","chapter":"Cell Structure","question":"Which organelle is known as the powerhouse of the cell?","answer":"Mitochondria"},
    {"class":"11","chapter":"Cell Structure","question":"Which organelle contains hydrolytic enzymes for intracellular digestion?","answer":"Lysosome"},
    {"class":"11","chapter":"Cell Structure","question":"Which organelle is the site of protein synthesis?","answer":"Ribosome"},
    {"class":"11","chapter":"Cell Structure","question":"Which structure regulates movement of substances into and out of the cell?","answer":"Plasma membrane"},
    {"class":"11","chapter":"Cell Structure","question":"What is the semifluid substance inside the cell that suspends organelles?","answer":"Cytoplasm"},
    {"class":"11","chapter":"Cell Structure","question":"Name the organelle responsible for photosynthesis in plants.","answer":"Chloroplast"},
    {"class":"11","chapter":"Cell Structure","question":"Which cytoskeletal elements provide shape and intracellular transport?","answer":"Microtubules and microfilaments"},
    {"class":"11","chapter":"Cell Structure","question":"What is the function of the Golgi apparatus?","answer":"Modification, sorting, and packaging of proteins and lipids"},
    {"class":"11","chapter":"Cell Structure","question":"What is the nucleolus mainly responsible for?","answer":"Ribosomal RNA synthesis and ribosome assembly"},

    # ----- Class 11: Biomolecules -----
    {"class":"11","chapter":"Biomolecules","question":"What are the monomers of proteins?","answer":"Amino acids"},
    {"class":"11","chapter":"Biomolecules","question":"Which bond links amino acids in proteins?","answer":"Peptide bond"},
    {"class":"11","chapter":"Biomolecules","question":"What are the monomers of nucleic acids?","answer":"Nucleotides"},
    {"class":"11","chapter":"Biomolecules","question":"Which polysaccharide stores energy in animals?","answer":"Glycogen"},
    {"class":"11","chapter":"Biomolecules","question":"Which polysaccharide stores energy in plants?","answer":"Starch"},
    {"class":"11","chapter":"Biomolecules","question":"Which polysaccharide is a major component of plant cell walls?","answer":"Cellulose"},
    {"class":"11","chapter":"Biomolecules","question":"What is the primary structure of a protein?","answer":"Unique sequence of amino acids"},
    {"class":"11","chapter":"Biomolecules","question":"Which lipids are the main components of membranes?","answer":"Phospholipids"},
    {"class":"11","chapter":"Biomolecules","question":"Which nitrogenous bases are purines?","answer":"Adenine and Guanine"},
    {"class":"11","chapter":"Biomolecules","question":"Which nitrogenous bases are pyrimidines?","answer":"Cytosine, Thymine, and Uracil"},

    # ----- Class 11: Enzymes -----
    {"class":"11","chapter":"Enzymes","question":"What is the active site of an enzyme?","answer":"Region where substrate binds and reaction occurs"},
    {"class":"11","chapter":"Enzymes","question":"How does temperature above optimum affect an enzyme?","answer":"Denaturation leading to loss of activity"},
    {"class":"11","chapter":"Enzymes","question":"How does pH affect enzyme activity?","answer":"Deviation from optimum alters ionization and reduces activity"},
    {"class":"11","chapter":"Enzymes","question":"What is a cofactor?","answer":"A non-protein helper (metal ion or coenzyme) required for activity"},
    {"class":"11","chapter":"Enzymes","question":"What is competitive inhibition?","answer":"Inhibitor competes with substrate for the active site"},
    {"class":"11","chapter":"Enzymes","question":"What is noncompetitive inhibition?","answer":"Inhibitor binds at a site other than active site reducing Vmax"},

    # ----- Class 11: Cell Division -----
    {"class":"11","chapter":"Cell Division","question":"Name the two main stages of the cell cycle.","answer":"Interphase and M phase"},
    {"class":"11","chapter":"Cell Division","question":"Which process produces two genetically identical daughter cells?","answer":"Mitosis"},
    {"class":"11","chapter":"Cell Division","question":"Which process reduces chromosome number by half?","answer":"Meiosis"},
    {"class":"11","chapter":"Cell Division","question":"During which phase do chromosomes align at the equator?","answer":"Metaphase"},
    {"class":"11","chapter":"Cell Division","question":"Crossing over occurs in which stage of meiosis?","answer":"Prophase I"},
    {"class":"11","chapter":"Cell Division","question":"What is cytokinesis?","answer":"Division of cytoplasm into two daughter cells"},
    {"class":"11","chapter":"Cell Division","question":"Name the protein structures that pull chromatids apart.","answer":"Spindle fibers (microtubules)"},
    {"class":"11","chapter":"Cell Division","question":"In which mitotic phase do sister chromatids separate?","answer":"Anaphase"},

    # ----- Class 11: Biological Diversity -----
    {"class":"11","chapter":"Diversity of Life","question":"Who proposed binomial nomenclature?","answer":"Carl Linnaeus"},
    {"class":"11","chapter":"Diversity of Life","question":"What is taxonomy?","answer":"Science of classification, identification, and naming of organisms"},
    {"class":"11","chapter":"Diversity of Life","question":"Name the five-kingdom classification proposer.","answer":"R.H. Whittaker"},
    {"class":"11","chapter":"Diversity of Life","question":"What is a species in biological terms?","answer":"A group of interbreeding natural populations reproductively isolated from others"},

    # ----- Class 11: Plant Anatomy & Physiology -----
    {"class":"11","chapter":"Plant Anatomy","question":"Which tissue transports water in plants?","answer":"Xylem"},
    {"class":"11","chapter":"Plant Anatomy","question":"Which tissue transports food in plants?","answer":"Phloem"},
    {"class":"11","chapter":"Plant Anatomy","question":"Which meristem increases length of plant organs?","answer":"Apical meristem"},
    {"class":"11","chapter":"Plant Anatomy","question":"What protects the tip of the root?","answer":"Root cap"},
    {"class":"11","chapter":"Plant Physiology","question":"Stomata regulate exchange of which gases?","answer":"CO2 and O2 (and water vapor)"},
    {"class":"11","chapter":"Plant Physiology","question":"What is transpiration?","answer":"Loss of water vapor from aerial parts of plants"},
    {"class":"11","chapter":"Plant Physiology","question":"Primary light-absorbing pigment in plants?","answer":"Chlorophyll a"},
    {"class":"11","chapter":"Plant Physiology","question":"Where does the Calvin cycle occur?","answer":"Stroma of chloroplast"},
    {"class":"11","chapter":"Plant Physiology","question":"Primary electron donor in photosystem II?","answer":"Water (H2O)"},
    {"class":"11","chapter":"Plant Physiology","question":"What is photophosphorylation?","answer":"Synthesis of ATP using light energy in chloroplasts"},

    # ----- Class 12: Homeostasis, Blood & Immunity -----
    {"class":"12","chapter":"Homeostasis","question":"Define homeostasis.","answer":"Maintenance of a stable internal environment"},
    {"class":"12","chapter":"Homeostasis","question":"Which organ secretes insulin?","answer":"Pancreas (beta cells)"},
    {"class":"12","chapter":"Homeostasis","question":"Which hormone increases blood glucose?","answer":"Glucagon"},
    {"class":"12","chapter":"Blood & Immunity","question":"What is the normal pH of human blood?","answer":"Approximately 7.4"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which cells transport oxygen in blood?","answer":"Red blood cells (erythrocytes)"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which blood component is essential for clotting?","answer":"Platelets (thrombocytes)"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which WBCs produce antibodies?","answer":"B lymphocytes (plasma cells)"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which protein in RBCs binds oxygen?","answer":"Hemoglobin"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which blood group is universal donor?","answer":"O negative"},
    {"class":"12","chapter":"Blood & Immunity","question":"Which blood group is universal recipient?","answer":"AB positive"},

    # ----- Class 12: Respiration & Excretion -----
    {"class":"12","chapter":"Respiration","question":"Where does gaseous exchange occur in lungs?","answer":"Alveoli"},
    {"class":"12","chapter":"Respiration","question":"Define tidal volume.","answer":"Volume of air inhaled or exhaled in a normal breath"},
    {"class":"12","chapter":"Respiration","question":"Name the pigment carrying oxygen in blood.","answer":"Hemoglobin"},
    {"class":"12","chapter":"Respiration","question":"What is vital capacity?","answer":"Maximum amount of air expelled after maximum inspiration"},
    {"class":"12","chapter":"Excretion","question":"Functional unit of kidney?","answer":"Nephron"},
    {"class":"12","chapter":"Excretion","question":"Where does filtration occur in the nephron?","answer":"Glomerulus (Bowman's capsule)"},
    {"class":"12","chapter":"Excretion","question":"What is reabsorption in the nephron?","answer":"Return of useful substances from filtrate to blood"},
    {"class":"12","chapter":"Excretion","question":"Where is ADH produced and what is its role?","answer":"Produced by hypothalamus; increases water reabsorption in kidneys"},
    {"class":"12","chapter":"Excretion","question":"What is the main nitrogenous waste in humans?","answer":"Urea"},
    {"class":"12","chapter":"Excretion","question":"Which part of nephron creates osmotic gradient?","answer":"Loop of Henle"},

    # ----- Class 12: Coordination & Control -----
    {"class":"12","chapter":"Coordination & Control","question":"Which brain part controls balance and coordination?","answer":"Cerebellum"},
    {"class":"12","chapter":"Coordination & Control","question":"Which division controls voluntary actions?","answer":"Somatic nervous system"},
    {"class":"12","chapter":"Coordination & Control","question":"Neurotransmitter at neuromuscular junction?","answer":"Acetylcholine"},
    {"class":"12","chapter":"Coordination & Control","question":"Which part of the brain regulates breathing and heart rate?","answer":"Medulla oblongata"},
    {"class":"12","chapter":"Coordination & Control","question":"Which lobe of brain is primarily for vision?","answer":"Occipital lobe"},
    {"class":"12","chapter":"Coordination & Control","question":"Which cells form myelin in the CNS?","answer":"Oligodendrocytes"},
    {"class":"12","chapter":"Coordination & Control","question":"Which cells form myelin in the PNS?","answer":"Schwann cells"},
    {"class":"12","chapter":"Coordination & Control","question":"Which ion triggers synaptic vesicle fusion?","answer":"Calcium (Ca2+)"},

    # ----- Class 12: Endocrine System -----
    {"class":"12","chapter":"Endocrine System","question":"Which gland is called the master gland?","answer":"Pituitary gland"},
    {"class":"12","chapter":"Endocrine System","question":"Which hormone regulates basal metabolic rate?","answer":"Thyroxine (T4)"},
    {"class":"12","chapter":"Endocrine System","question":"Which gland secretes adrenaline?","answer":"Adrenal medulla"},
    {"class":"12","chapter":"Endocrine System","question":"Hormone responsible for calcium regulation by lowering blood Ca2+?","answer":"Calcitonin"},
    {"class":"12","chapter":"Endocrine System","question":"Which hormone increases blood calcium levels?","answer":"Parathyroid hormone (PTH)"},
    {"class":"12","chapter":"Endocrine System","question":"Which hormone is antidiuretic?","answer":"ADH (vasopressin)"},
    {"class":"12","chapter":"Endocrine System","question":"Which pancreatic cells secrete glucagon?","answer":"Alpha cells"},

    # ----- Class 12: Reproduction -----
    {"class":"12","chapter":"Reproduction","question":"Define fertilization.","answer":"Fusion of male and female gametes"},
    {"class":"12","chapter":"Reproduction","question":"Define implantation.","answer":"Attachment of the embryo to the uterine wall"},
    {"class":"12","chapter":"Reproduction","question":"What is placenta?","answer":"Organ for exchange of nutrients, gases, and wastes between mother and fetus"},
    {"class":"12","chapter":"Reproduction","question":"Where are Leydig cells located and what do they secrete?","answer":"In testes; secrete testosterone"},
    {"class":"12","chapter":"Reproduction","question":"Where does oogenesis occur?","answer":"Ovaries"},
    {"class":"12","chapter":"Reproduction","question":"Which hormone triggers ovulation?","answer":"LH (Luteinizing Hormone)"},

    # ----- Class 12: Genetics -----
    {"class":"12","chapter":"Genetics","question":"Who is the father of genetics?","answer":"Gregor Mendel"},
    {"class":"12","chapter":"Genetics","question":"What is phenotype?","answer":"Observable characteristics of an organism"},
    {"class":"12","chapter":"Genetics","question":"What is genotype?","answer":"Genetic makeup of an organism"},
    {"class":"12","chapter":"Genetics","question":"What are alleles?","answer":"Alternative forms of a gene"},
    {"class":"12","chapter":"Genetics","question":"What is a test cross?","answer":"Cross with homozygous recessive to determine genotype"},
    {"class":"12","chapter":"Genetics","question":"Which principle explains separation of allele pairs?","answer":"Law of Segregation"},
    {"class":"12","chapter":"Genetics","question":"Which inheritance shows blending of traits?","answer":"Incomplete dominance"},
    {"class":"12","chapter":"Genetics","question":"Which process makes mRNA from DNA?","answer":"Transcription"},
    {"class":"12","chapter":"Genetics","question":"Which process synthesizes protein from mRNA?","answer":"Translation"},
    {"class":"12","chapter":"Genetics","question":"Which enzyme synthesizes RNA from a DNA template?","answer":"RNA polymerase"},

    # ----- Class 12: Evolution -----
    {"class":"12","chapter":"Evolution","question":"Define natural selection.","answer":"Differential survival and reproduction of individuals due to heritable traits"},
    {"class":"12","chapter":"Evolution","question":"What is speciation?","answer":"Formation of new species"},
    {"class":"12","chapter":"Evolution","question":"What are homologous structures?","answer":"Structures with similar architecture indicating common ancestry"},
    {"class":"12","chapter":"Evolution","question":"What are analogous structures?","answer":"Structures with similar function but different origin"},
    {"class":"12","chapter":"Evolution","question":"What is genetic drift?","answer":"Random change in allele frequencies in small populations"},

    # ----- Ecology -----
    {"class":"12","chapter":"Ecology","question":"Define ecosystem.","answer":"Community of organisms interacting with their physical environment"},
    {"class":"12","chapter":"Ecology","question":"What is a trophic level?","answer":"Position of an organism in a food chain"},
    {"class":"12","chapter":"Ecology","question":"Who are producers?","answer":"Autotrophs that synthesize organic compounds"},
    {"class":"12","chapter":"Ecology","question":"Define food chain.","answer":"Linear sequence of organisms through which nutrients and energy pass"},
    {"class":"12","chapter":"Ecology","question":"What is nitrogen fixation?","answer":"Conversion of atmospheric nitrogen into ammonia"},

    # ----- Biotechnology / MDCAT Quick -----
    {"class":"12","chapter":"Biotechnology","question":"What is genetic engineering?","answer":"Direct manipulation of an organism's DNA"},
    {"class":"12","chapter":"Biotechnology","question":"Which enzymes cut DNA at specific sequences?","answer":"Restriction endonucleases"},
    {"class":"12","chapter":"Biotechnology","question":"What technique amplifies DNA segments?","answer":"Polymerase Chain Reaction (PCR)"},
    {"class":"MDCAT","chapter":"Quick","question":"Where does glycolysis occur?","answer":"Cytoplasm"},
    {"class":"MDCAT","chapter":"Quick","question":"Where does Krebs cycle occur?","answer":"Mitochondrial matrix"},
    {"class":"MDCAT","chapter":"Quick","question":"Gas used by plants in photosynthesis?","answer":"Carbon dioxide (CO2)"},
    {"class":"MDCAT","chapter":"Quick","question":"Sugar formed in photosynthesis?","answer":"Glucose"},
    {"class":"MDCAT","chapter":"Quick","question":"Which vitamin is synthesized in skin by sunlight?","answer":"Vitamin D"},
    {"class":"MDCAT","chapter":"Quick","question":"Largest organ of human body?","answer":"Skin"},
    {"class":"MDCAT","chapter":"Quick","question":"Hormone for fight-or-flight response?","answer":"Adrenaline (epinephrine)"},
]

base_df = pd.DataFrame(MCQS, columns=["class","chapter","question","answer"])

with st.expander("Upload more MCQs (optional)"):
    st.write("CSV columns required: class, chapter, question, answer")
    up = st.file_uploader("Upload CSV to merge", type=["csv"])
    merged_df = base_df.copy()
    if up:
        try:
            ext = pd.read_csv(up)
            ext.columns = [c.strip().lower() for c in ext.columns]
            needed = {"class","chapter","question","answer"}
            if not needed.issubset(set(ext.columns)):
                st.error("CSV must contain columns: class, chapter, question, answer")
            else:
                ext = ext[list(needed)]
                merged_df = pd.concat([merged_df, ext], ignore_index=True)
                st.success(f"Merged! Total rows: {len(merged_df)}")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    else:
        st.info(f"Using built-in dataset: {len(base_df)} rows")

with st.expander("Filter"):
    c1, c2 = st.columns(2)
    classes = ["All"] + sorted(merged_df["class"].astype(str).unique().tolist())
    chapters = ["All"] + sorted(merged_df["chapter"].astype(str).unique().tolist())
    sel_class = c1.selectbox("Class", classes, index=0)
    sel_chap = c2.selectbox("Chapter", chapters, index=0)

df = merged_df.copy()
if sel_class != "All":
    df = df[df["class"].astype(str) == str(sel_class)]
if sel_chap != "All":
    df = df[df["chapter"].astype(str) == str(sel_chap)]

st.divider()
query = st.text_input("Type your question:", placeholder="e.g., Where does glycolysis occur?")

def retrieve_answer(q, data, topk=5):
    if data.empty or not q.strip():
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(data["question"].astype(str).fillna("").tolist())
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:topk]
    out = []
    for i in idxs:
        out.append((float(sims[i]), data.iloc[i]))
    return out

if query:
    results = retrieve_answer(query, df, topk=5)
    if results and results[0][0] > 0.12:
        score, row = results[0]
        st.success(f"Answer: {row['answer']}")
        st.caption(f"Match: {score:.2f} • Class {row['class']} • {row['chapter']}")
        with st.expander("Related Q&A"):
            for sc, r in results:
                st.markdown(f"- **Q:** {r['question']}  \n  **A:** {r['answer']}  \n  _Match {sc:.2f} | Class {r['class']} | {r['chapter']}_")
    else:
        st.error("No close match found. Try rephrasing.")
else:
    st.info("Enter a Biology question above.")

st.divider()
csv_buf = StringIO()
merged_df.to_csv(csv_buf, index=False)
st.download_button("Download Merged CSV (bio_qa.csv)", data=csv_buf.getvalue(), file_name="bio_qa.csv", mime="text/csv")
