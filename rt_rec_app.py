import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle 
from fuzzywuzzy import fuzz



# main function
def main():
	# create a title
	st.title('Tomatoes Recommender -- Discover Some New Favs')

	# set the path to the file containing the index titles
	INDEX_PATH = 'index_titles.txt'
	# set the path to the feature matrix
	MATRIX_PATH = 'final_feat_matrix.npz'
	# set the path for the link dictionary
	LINKDICT_PATH = 'links_dict.txt'
	# and the number of neighors/recs
	NN = 10

	# load in the hashmap to map titles to index values in the feature matrix
	with open(INDEX_PATH, 'rb') as f:
		index_titles = pickle.load(f)

	hash_map = {title:i for i, title in enumerate(index_titles)}

	with open(LINKDICT_PATH, 'rb') as fp:
		links_dict = pickle.load(fp)

	# define a function for taking in a title and outputting recs
	def get_recs(input_title):

		# look up idx of selection in the hashmap
		movie_idx = hash_map[input_title]

		#  load and fit model (load in feat matrix first)
		feat_matrix = csr_matrix(load_npz(MATRIX_PATH))
		model = NearestNeighbors(metric='cosine', n_neighbors=NN)
		model.fit(feat_matrix)


		# 3 -- find distances with model
		distances, indices = model.kneighbors(feat_matrix[movie_idx], n_neighbors=NN+1)

		# 4 -- convert indices and distances into sorte list
		raw_recs = sorted(
		                list(
		                    zip(
		                        indices.squeeze().tolist(),
		                        distances.squeeze().tolist()
		                  )
		), 
		key=lambda x: x[1]
		)[1:]

		# reverse the hashmap
		reverse_hashmap = {v:k for k, v in hash_map.items()}

		# get the titles recommended
		title_recs = [(str(reverse_hashmap[v])) for v, k in raw_recs]

		input_link = links_dict[input_title]
		st.subheader(f'Top Recommendations for: [{input_title}]({input_link})')

		for item in title_recs:
			link = links_dict[item]
			st.write(f"[{item}]({link})")


	# initialize the session_state property to keep the input title
	welcome_message = 'Enter a title to get started!'
	if 'input_title' not in st.session_state:
		st.session_state['input_title'] = welcome_message

	# if already looped through once, display recommendations
	if st.session_state.input_title != welcome_message:
		get_recs(st.session_state.input_title)
	else:
		st.subheader(st.session_state.input_title)

	# 2 -- get an input from user
	with st.form(key='my_form'):
	    user_input = st.text_input(label='Enter a movie title')
	    submit_button = st.form_submit_button(label='Submit')


	# on button submission
	if submit_button:
		# empty list to capture candidate titles
		matches = []
		# loop through hash_map now
		for title, idx in hash_map.items():
			ratio = fuzz.ratio(title.lower(), user_input.lower())
			if ratio > 60:
				matches.append((title, idx, ratio))

		# sort list
		matches = sorted(matches, key=lambda x: x[2])[::-1] # to reverse list?
		matches = [item[0] for item in matches]

		if not matches:
			st.write('No matches found!')

		else:
			# define a callback function for the selectbox
			def update_recs():
				st.session_state.input_title = st.session_state.movie_selection
			selection = st.selectbox(label='Closet matches are: ',
								   options=['See options']+matches,
								   key = 'movie_selection',
								   on_change=update_recs
								   )
		

main()