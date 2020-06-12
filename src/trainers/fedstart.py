from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.prox_gd import START
import numpy as np
import torch


criterion = torch.nn.CrossEntropyLoss()


class FedSTART(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = START(model.parameters(), lr=options['lr'], weight_decay=options['wd'], mu=options['mu'])
        self.num_epoch = options['num_epoch']
        worker = LrdWorker(model, self.optimizer, options)
        super(FedSTART, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.adjust_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= self.all_train_data_num
        averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()
