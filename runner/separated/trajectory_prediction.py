import numpy as np
import os
import torch
import pickle
from st_prediction.models.model import Informer
from st_prediction.myutils.tools import StandardScaler
from st_prediction.exp.data_loader import Dataset_Pred
from torch.utils.data import DataLoader
from scipy.spatial import distance


class InformerModule:
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, factor, d_model, n_heads, e_layers, d_layers,
                       d_ff, dropout, attn, embed, freq, activation, output_attention, distil, mix, device):

        self.model = Informer(enc_in=enc_in, dec_in=dec_in, c_out=c_out, seq_len=seq_len, label_len=label_len,
                     out_len=pred_len, factor=factor, d_model=d_model, n_heads=n_heads, e_layers=e_layers,
                     d_layers=d_layers, d_ff=d_ff, dropout=dropout, attn=attn, embed=embed, freq=freq,
                     activation=activation, output_attention=output_attention, distil=distil, mix=mix, device=device).float().to(device)
        self.scaler = StandardScaler()
        self.device = device

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

    def load(self, setting):
        model_path = os.path.join("st_prediction/checkpoints", setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(model_path))
        scaler_path = os.path.join("st_prediction/checkpoints", setting, 'scaler.npz')
        parameters = np.load(scaler_path, allow_pickle=True)
        self.scaler.load(mean=parameters["mean"], std=parameters["std"])

    def get_data(self, seqs_x, seqs_x_mark, seqs_y, seqs_y_mark):
        data_set = Dataset_Pred(
            scaler=self.scaler,
            seqs_x=seqs_x,
            seqs_x_mark=seqs_x_mark,
            seqs_y=seqs_y,
            seqs_y_mark=seqs_y_mark
        )
        data_loader = DataLoader(
            data_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False)
        return data_loader

    def predict(self, edge_pos, seqs_x, seqs_x_mark, seqs_y, seqs_y_mark):
        pred_loader = self.get_data(seqs_x, seqs_x_mark, seqs_y, seqs_y_mark)
        self.model.eval()
        preds = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        outputs = []
        # 根据坐标匹配最近的服务器
        for i in range(np.shape(preds)[0]):
            output = []
            for j in range(np.shape(preds)[1]):
                vehicle_pos = preds[i][j]
                dis_list = []
                for index, edge_i in edge_pos.iterrows():
                    edge_i_pos = [edge_i["edge_pos_x"], edge_i["edge_pos_y"]]
                    dist_i = distance.cdist(np.array(edge_i_pos).reshape(1, -1), np.array(vehicle_pos).reshape(1, -1),
                                            metric='euclidean')[0][0]
                    dis_list.append(dist_i)
                min_index = np.argmin(dis_list)
                output.append(edge_pos.iloc[min_index]["edge_id"])
            outputs.append(output)
        return outputs

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # encoder - decoder
        if self.output_attention:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        outputs = self.scaler.inverse_transform(outputs)
        return outputs


class Trajectory_Loader:
    def __init__(self, time_and_trajectory):
        self.vehicle_start_time = time_and_trajectory["start_time"]
        self.history_time_features = time_and_trajectory["time_features"]
        self.history_trajectory = time_and_trajectory["trajectory"]

    def query_history(self, current_time, seq_len, label_len, pred_len):
        flag = 0
        if ((current_time + 1 - seq_len) < self.vehicle_start_time) or \
                ((self.vehicle_start_time + np.shape(self.history_trajectory)[0]-1) < current_time):
            seq_x, seq_x_mark, seq_y, seq_y_mark = None, None, None, None
        else:
            flag = 1  # 数据有效
            index_time = current_time - self.vehicle_start_time
            x_begin = int(index_time+1-seq_len)
            x_end = int(index_time+1)
            y_begin = x_end - label_len

            seq_x = self.history_trajectory[x_begin:x_end]
            seq_x_mark = self.history_time_features[x_begin:x_end]
            seq_y = np.concatenate([self.history_trajectory[y_begin:x_end],
                                    np.zeros([pred_len, self.history_trajectory.shape[-1]])], axis=0)
            seq_y_mark = np.concatenate([self.history_time_features[y_begin:x_end],
                                         np.zeros([pred_len, self.history_time_features.shape[-1]])], axis=0)
        return flag, seq_x, seq_x_mark, seq_y, seq_y_mark
