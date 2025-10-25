from sklearn.model_selection import cross_validate, train_test_split
from hgp_lib.preprocessing import binarize
from hgp_lib.algorithm import HierarchicalGP
from hgp_lib.mutation import MutationExecutor

train_test_split

data = ...
labels = ...

(train_data, train_labels), (test_data, test_labels) = train_test_split(data, labels)


# OPTIONAL

(train_data, train_labels), (val_data, val_labels) = train_test_split(train_data, train_labels)

# TODO: Maybe we should use sklearn for binarize, because it can be fitted only on train, and for the rest we use transform
train_data = binarize(train_data)
val_data = binarize(val_data)
test_data = binarize(test_data)

list_of_mutations = ...



mutation_executor = MutationExecutor(list_of_mutations)

num_iterations = 1000

gp_algo = HierarchicalGP(mutation_executor)

for i in range(num_iterations):
    train_metrics = gp_algo.step(train_data, train_labels)

    if i % 100 == 0:
        val_metrics = gp_algo.validate(val_data, val_labels)



trainer = Trainer(
    num_iterations,
    val_every_step,
    mutation_list,
)


trainer.run()


benchmarker = Benchmarker(

)




