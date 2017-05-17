# coding:utf8
from keras.models import Sequential, Model
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

if K.backend() == "tensorflow":
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    K.set_session(session)


with open("D:/DataForMining/sth/all.txt") as f:
    all_data = [line.strip().split(";") for line in f]
data_x_team_id_raw = all_data[0::9]
data_y_raw = all_data[1::9]
data_x_team_abbr_raw = all_data[2::9]
data_x_home_min_raw = all_data[3::9]
data_x_home_id_raw = all_data[4::9]
data_x_home_name_raw = all_data[5::9]
data_x_visitor_min_raw = all_data[6::9]
data_x_visitor_id_raw = all_data[7::9]
data_x_visitor_name_raw = all_data[8::9]


def flatten(x):
    for seq in x:
        for s in seq:
            yield s

# 简单来说对应去重打包
id2team = dict(zip(flatten(data_x_team_id_raw), flatten(data_x_team_abbr_raw)))
id2player = dict(zip(flatten(data_x_home_id_raw + data_x_visitor_id_raw),
                     flatten(data_x_home_name_raw + data_x_visitor_name_raw)))
team2id = dict(zip(flatten(data_x_team_abbr_raw), flatten(data_x_team_id_raw)))
player2id = dict(zip(flatten(data_x_home_name_raw + data_x_visitor_name_raw),
                     flatten(data_x_home_id_raw + data_x_visitor_id_raw)))

print("total_teams", len(id2team))
print("total_players", len(id2player))

# 训练集序列化
tid2index = {tid: idx for idx, tid in enumerate(id2team)}
index2tid = {idx: tid for idx, tid in enumerate(id2team)}

pid2index = {pid: idx + 1 for idx, pid in enumerate(id2player)}
index2pid = {idx + 1: pid for idx, pid in enumerate(id2player)}


def str2time(ms):
    m, s = ms.split(":")
    return int(m) * 60 + int(s)

# 训练集id化
data_x_team_id = np.array(
    list(map(lambda x: [tid2index[tid] for tid in x], data_x_team_id_raw)))

data_x_home_id = pad_sequences(list(map(lambda x: [
                               pid2index[pid] for pid in x], data_x_home_id_raw)), padding="post", maxlen=13)

data_x_vistor_id = pad_sequences(list(map(lambda x: [
                                 pid2index[pid] for pid in x], data_x_visitor_id_raw)), padding="post", maxlen=13)

data_x_home_min = pad_sequences(list(map(lambda x: [str2time(
    ms) for ms in x], data_x_home_min_raw)), padding="post", maxlen=13)

data_x_visitor_min = pad_sequences(list(map(lambda x: [str2time(
    ms) for ms in x], data_x_visitor_min_raw)), padding="post", maxlen=13)

# 时间归一化
data_x_home_min = 5 * \
    data_x_home_min.astype(K.floatx()) / data_x_home_min.sum(axis=-1)[:, None]

data_x_visitor_min = 5 * \
    data_x_visitor_min.astype(
        K.floatx()) / data_x_visitor_min.sum(axis=-1)[:, None]

data_y = np.array(data_y_raw, dtype=int)


######################
# 构造比分预测模型：
x_t = Input((2,), name="team_id")

x_h_id = Input((13,), name="home_player_id")
x_h_min = Input((13, 1), name="home_player_time")

x_v_id = Input((13,), name="visitor_player_id")
x_v_min = Input((13, 1), name="visitor_player_time")

emb_dim = 256

team_emb = Sequential(name="team_emb")
team_emb.add(Embedding(input_dim=30, output_dim=emb_dim, input_length=2))
team_emb.add(Flatten())

player_emb = Sequential(name="player_emb")
player_emb.add(
    Embedding(input_dim=len(id2player) + 1, output_dim=emb_dim, input_length=13))

feat_t = team_emb(x_t)
print(feat_t)
print("======")

feat_h_id = player_emb(x_h_id)
feat_h = merge([x_h_min, feat_h_id], mode='dot', dot_axes=1)
feat_h = Reshape((emb_dim, ), name="home_feat")(feat_h)
print(feat_h)
print("======")

feat_v_id = player_emb(x_v_id)
feat_v = merge([x_v_min, feat_v_id], mode='dot', dot_axes=1)
feat_v = Reshape((emb_dim, ), name="visitor_feat")(feat_v)
print(feat_v)
print("======")

feat = merge([feat_h, feat_v, feat_t], mode='concat', concat_axis=0)


hid = Dense(256, activation="relu", name="hidden_1")(feat)
hid = Dropout(0.2, name="dropout_1")(hid)
hid = Dense(128, activation="relu", name="hidden_2")(hid)
hid = Dropout(0.2, name="dropout_2")(hid)
score = Dense(2, activation="relu", name="score")(hid)


# model
model = Model([x_t, x_h_id, x_h_min, x_v_id, x_v_min], score)


def win(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_true, axis=-1)))

model.compile(optimizer="adam", loss="mae", metrics=[win])

np.random.seed(1105)

idx = np.arange(len(data_y))
np.random.shuffle(idx)

train_idx = idx[600:]
valid_idx = idx[:600]

print(len(idx))

hist = model.fit([data_x_team_id[train_idx],
                  data_x_home_id[train_idx], data_x_home_min[
                      train_idx, :, None],
                  data_x_vistor_id[train_idx], data_x_visitor_min[train_idx, :, None]],
                 data_y[train_idx],
                 validation_data=([data_x_team_id[valid_idx],
                                   data_x_home_id[valid_idx], data_x_home_min[
                     valid_idx, :, None],
                     data_x_vistor_id[valid_idx], data_x_visitor_min[valid_idx, :, None]],
    data_y[valid_idx]),
    verbose=0,
    nb_epoch=30)

hist.history["val_loss"]

data_t = model.predict([data_x_team_id[valid_idx],
                        data_x_home_id[valid_idx], data_x_home_min[
                            valid_idx, :, None],
                        data_x_vistor_id[valid_idx], data_x_visitor_min[valid_idx, :, None]])

print(np.mean(data_t.argmax(axis=-1) == data_y[valid_idx].argmax(axis=-1)))
print(np.mean(0 == data_y[valid_idx].argmax(axis=-1)))

# 测试集上的预测比分：
print(data_t)
print("==========")
# 真实比分：
print(
    data_y[valid_idx])

data_x_team_abbr = np.array(data_x_team_abbr_raw)
print(data_x_team_abbr[valid_idx])
# 训练集上的比分和队伍：
data_p = model.predict([data_x_team_id[train_idx],
                        data_x_home_id[train_idx], data_x_home_min[
                            train_idx, :, None],
                        data_x_vistor_id[train_idx], data_x_visitor_min[train_idx, :, None]])

print(data_p)
print(
    data_x_team_abbr[train_idx])
print(data_y[train_idx])


####################
# 模型结构
# Keras可视化 以后在折腾
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot, plot_model
#
# plot_model(model, to_file="model.png")
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
