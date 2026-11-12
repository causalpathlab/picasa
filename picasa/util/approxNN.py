import annoy
from anndata import AnnData
import logging
import random
logger = logging.getLogger(__name__)

class ApproxNN():
	def __init__(self, data):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist()[0])
		self.index.build(number_of_trees)

	def query(self, vector, k):
		indexes, distances = self.index.get_nns_by_vector(vector.tolist()[0], k, include_distances=True)
		return list(zip(indexes, distances))


def get_NNmodel(mtx,number_of_trees):
    model_ann = ApproxNN(mtx)
    model_ann.build(number_of_trees)
    return model_ann
	 


def get_neighbours(mtx, model, nbrsize=1, strategy='first'):
    nbr_dict = {}
    total_items = model.index.get_n_items()
    
    for idx, row in enumerate(mtx):
        if strategy == 'first':
            res = model.query(row, k=nbrsize)
            nbr_dict[idx] = res[0]
        elif strategy == 'random':
            # Pick a random index and assign a dummy distance of 0.0
            rand_idx = random.randint(0, total_items - 1)
            nbr_dict[idx] = (rand_idx, 0.0)
        else:
            # Query all items to get full sorted distance ranking
            res = model.query(row, k=total_items)
            if strategy == 'last':
                nbr_dict[idx] = res[-1]
            elif strategy == 'middle':
                mid_idx = len(res) // 2
                nbr_dict[idx] = res[mid_idx]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
    return nbr_dict

def generate_neighbours(source_adata: AnnData, 
                        target_adata: AnnData,
                        tag: str,
                        number_of_trees: int = 50 
                        ) -> dict:
    ann_model = get_NNmodel(source_adata.X.todense(), number_of_trees)
    nbr_dict = get_neighbours(target_adata.X.todense(), ann_model, nbrsize=1, strategy='first')
    return nbr_dict


def generate_neighbours_last(source_adata: AnnData, 
                             target_adata: AnnData,
                             tag: str,
                             number_of_trees: int = 50 
                             ) -> dict:
    ann_model = get_NNmodel(source_adata.X.todense(), number_of_trees)
    # Querying with total items to extract the farthest point
    total_items = source_adata.shape[0]
    nbr_dict = get_neighbours(target_adata.X.todense(), ann_model, nbrsize=total_items, strategy='last')
    return nbr_dict


def generate_neighbours_middle(source_adata: AnnData, 
                               target_adata: AnnData,
                               tag: str,
                               number_of_trees: int = 50 
                               ) -> dict:
    ann_model = get_NNmodel(source_adata.X.todense(), number_of_trees)
    total_items = source_adata.shape[0]
    nbr_dict = get_neighbours(target_adata.X.todense(), ann_model, nbrsize=total_items, strategy='middle')
    return nbr_dict

def generate_neighbours_random(source_adata: AnnData, 
                               target_adata: AnnData,
                               tag: str,
                               number_of_trees: int = 50 
                               ) -> dict:
    ann_model = get_NNmodel(source_adata.X.todense(), number_of_trees)
    total_items = source_adata.shape[0]
    nbr_dict = get_neighbours(target_adata.X.todense(), ann_model, nbrsize=total_items, strategy='random')
    return nbr_dict