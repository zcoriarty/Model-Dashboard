'''Get functions for accessing colors'''

def get_colors(data):
	colors_id = {'Hold': '#F2BE4A',
					'Buy': '#01A6A4',
					'Sell': '#EC6355'}

	return colors_id

def get_labels(data):
	label_id = {'0': 'Hold',
				'1': 'Buy', 
				'2': 'Sell'}
	labels = data['Choice'].map(label_id)

	return labels

def get_colorscale():

	# specifies values for gradient scale
	colorscale = [[0, '#ff7845'], # null values
					[.5, '#ce6453'], # hold
					[.75, '#5d9ea2'], # buy
					[1, '#248797']] # sell

	return colorscale



	
	
				
	
	

	