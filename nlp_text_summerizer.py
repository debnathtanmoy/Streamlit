import streamlit as st 

from gensim.summarization import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import spacy
from spacy import displacy
nlp = spacy.load(('en_core_web_sm'))

from bs4 import BeautifulSoup
from urllib.request import urlopen

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def main():

	st.title("Summaryzer and Entity Checker")

	activities = ["Summarize","NER Checker","NER For URL"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == 'Summarize':
		st.subheader("Summarize Document")
		raw_text = st.text_area("Enter Text Here","Type Here")
		summarizer_type = st.selectbox("Summarizer Type",["Gensim","Sumy Lex Rank"])
		if st.button("Summarize"):
			if summarizer_type == "Gensim":
				summary_result = summarize(raw_text)
			elif summarizer_type == "Sumy Lex Rank":
				summary_result = sumy_summarizer(raw_text)

			st.write(summary_result)

	
				
		


if __name__ == '__main__':
	main()