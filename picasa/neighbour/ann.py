import annoy
from anndata import AnnData
import logging
logger = logging.getLogger(__name__)

class ApproxNN():
	def __init__(self, data):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist())
		self.index.build(number_of_trees)

	def query(self, vector, k):
		indexes = self.index.get_nns_by_vector(vector.tolist(),k)
		return indexes


def get_NNmodel(mtx):
    model_ann = ApproxNN(mtx)
    model_ann.build()
    return model_ann

def get_neighbours(mtx,model,nbrsize):
		
	nbr_dict={}
	for idx,row in enumerate(mtx):
		nbr_dict[idx] = model.query(row,k=nbrsize)
	return nbr_dict

def generate_neighbours(source_adata: AnnData, 
                        target_adata: AnnData,
                        use_projection: str = 'X_rp',
                        num_nbrs: int = 1
                        ) -> dict:
    
    ann_model = get_NNmodel(source_adata.obsm[use_projection])
    nbr_dict = get_neighbours(target_adata.obsm[use_projection],ann_model,num_nbrs)
    return nbr_dict