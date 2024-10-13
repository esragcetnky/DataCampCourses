from transformers import pipeline

original_text = """\nGreece has many islands, with estimates ranging from somewhere around 1,200 to 6,000, 
                depending on the minimum size to take into account. The number of inhabited islands is variously 
                cited as between 166 and 227.\nThe Greek islands are traditionally grouped into the following clusters: 
                the Argo-Saronic Islands in the Saronic Gulf near Athens; the Cyclades, a large but dense collection occupying 
                he central part of the Aegean Sea; the North Aegean islands, a loose grouping off the west coast of Turkey; 
                the Dodecanese, another loose collection in the southeast between Crete and Turkey; the Sporades, a small tight group off 
                the coast of Euboea; and the Ionian Islands, chiefly located to the west of the mainland in the Ionian Sea. Crete with its 
                surrounding islets and Euboea are traditionally excluded from this grouping.\n"""

# Create a short summarizer
short_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_length=1, max_length=10)

# Summarize the input text
short_summary_text = short_summarizer(original_text)

# Repeat for a long summarizer
long_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", max_length=150, min_length=50)

long_summary_text = long_summarizer(original_text)

print("#################################################################################################")
# Print the short summary
print(short_summary_text[0]["summary_text"])
# Print the long summary
print(long_summary_text[0]["summary_text"])