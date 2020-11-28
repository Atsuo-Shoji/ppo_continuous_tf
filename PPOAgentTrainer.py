# -*- coding: utf-8 -*-
#Trainer
#環境envとAgentを持ち、Agentの訓練を受け持つ
#PPOに沿ってAgentをtrainする

import tensorflow as tf
import numpy as np
from datetime import datetime

from common.funcs import *


class PPOAgentTrainer():
    
    def __init__(self, name, env, agent):
        
        self._name = name
        self._env = env
        self._agent = agent

        self._state_dim = self._agent.state_dim
        self._action_dim = self._agent.action_dim
        
        #報酬の蓄積
        #Reward Scalingのための全報酬の標準偏差の算出に使用
        #経験バッファと異なり、エポックで洗い替えない（train開始時点から蓄積し続ける）
        #train()を連続して呼ばれることを考慮し、メンバ変数とする
        self._calc_stdev = PPOAgentTrainer.Calculater_Statistics.createInstance()
        
    def train(self, epochs, trajectory_size=1024, lamda_GAE=0.95, gamma=0.99, batch_size=1024, clip_range=0.2, 
              standardize_rewards_per_epoch=False, verbose_interval=1):
        
        start_time = datetime.now()
        
        #訓練結果の記録　エポック毎の記録
        loss_actor_epochs = [] #Actorのlossのエポック毎の記録　正確には、そのエポックでのNN更新前のloss
        loss_critic_epochs = [] #Criticのlossのエポック毎の記録　正確には、そのエポックでのNN更新前のloss
        steps_epochs = [] #NN更新後の1エピソードPlayでのステップ数のエポック毎の記録
        score_epochs = [] #NN更新後の1エピソードPlayでの稼得score（報酬合計）のエポック毎の記録
        best_score = -np.inf #NN更新後の1エピソードPlayでの稼得score　今までのエポックの中のベストscore
        best_score_count = 0 #NN更新後の1エピソードPlayでの稼得score　ベストscore更新回数
        
        #現時点での訓練対象パラメーターを一時退避　ベストscore達成によるパラメーター一時退避が起こらないこともありうる
        self._agent.keep_temporarily_learnable_params()
        
        #env初期化
        st = self._env.reset()
        #エピソード終端（done=True）で、next_stが無い（None）時のダミー
        #delta計算時にNoneだと計算不可になるので（あまり良くないが他にいい方法が思いつかない）
        next_st_zero_for_none = np.zeros_like(st, dtype=np.float32)

        for epc in range(epochs):
            
            #1エポック
            #経験データ（trajectory）収集
            #収集した全サンプルについて、GAEを算出
            #収集した全サンプルとGAEを使用して、Agent（ActorとCritic）を更新
            
            ##経験データ（trajectory）収集##
            
            #経験バッファクリア　インスタンス生成
            #全データの特定の列に対して演算や置換を行うので、列の集合体とする
            trajectory = {"state": [], "action": [], "reward": [], "next_state": [], "done": [], "policy": [], 
                          "reward_raw": [], "GAE":[], "Vtarg":[]}
            
            #経験バッファサイズまでstepし、経験データを経験バッファに保存
            for d in range(trajectory_size):
                
                #agentに最適行動を推測させる
                #最適行動と、その最適行動の正規分布上での確率密度関数値が返される
                a, pi = self._agent.actor.predict_best_action_and_policy(st)
                #最適行動で1step進める
                next_st, rew, done, _  = self._env.step(a)
                
                if done==True:
                    next_st = next_st_zero_for_none
                
                #経験バッファに追加
                trajectory["state"].append(st)
                trajectory["action"].append(a)
                trajectory["reward_raw"].append(rew)
                trajectory["reward"].append(rew)
                trajectory["next_state"].append(next_st)
                trajectory["done"].append(done)
                trajectory["policy"].append(pi)
                
                #報酬の蓄積
                #reward_accumulated.append(rew)

                if done==True:
                    st = self._env.reset()
                else:
                    st = next_st

            #経験バッファ内の各列はListであるが、これだと列単位の演算に不向きなので、ndarrayに置換
            #reshapeは、統一的に配列のaxis=0をデータ数trajectory_sizeにするため。
            #ただし、trajectory_size>1なので、次元数が1より大きい変数（stateなど）は、reshapeによる変化は無い。
            #変化が有るのは、rewardのような本来はスカラー値の変数で、(trajectory_size,)→(trajectory_size, 1)にする。
            #プログラム仕様上、state_dimやaction_dimが1であることもありうるので、一応全部の変数について、統一的にreshapeする。
            #（Pendulumのaction_dimは1である）
            #注意！np.atleast_2dを使用しないこと！
            #np.atleast_2dは、「データ件数が1の場合」、統一して、配列のaxis=0をデータ件数に使用するためのshape変換。
            #つまり、shape(数,)の場合、(1, 数)と変換する。ここでは、(trajectory_size,)なら(1, trajectory_size)としてしまう。
            #しかし、今したいのは、(数,)→(数,1)の変換。そもそも「データ件数が1の場合」はありえない想定。
            trajectory["state"] = np.array(trajectory["state"], dtype=np.float32).reshape(-1, self._state_dim)
            trajectory["action"] = np.array(trajectory["action"], dtype=np.float32).reshape(-1, self._action_dim)
            trajectory["reward_raw"] = np.array(trajectory["reward_raw"], dtype=np.float32).reshape(-1, 1)
            trajectory["reward"] = np.array(trajectory["reward"], dtype=np.float32).reshape(-1, 1)
            trajectory["next_state"] = np.array(trajectory["next_state"], dtype=np.float32).reshape(-1, self._state_dim)
            trajectory["done"] = np.array(trajectory["done"], dtype=np.float32).reshape(-1, 1)
            trajectory["policy"] = np.array(trajectory["policy"], dtype=np.float32).reshape(-1, self._action_dim)
            
            #報酬をエポックごとに標準化する場合
            if standardize_rewards_per_epoch==True:
                #平均を0にしない標準化
                rewards_std = standardize(trajectory["reward_raw"], with_mean=False).reshape(-1, 1)
                trajectory["reward"] = rewards_std
            
            ##GAEの算出##

            #Reward Scalingのための蓄積報酬の標準偏差算出
            #sigma_reward_accumulated = np.std(reward_accumulated_arr)
            #↓squeeze()は、この時点でのshape(trajectory_size, 1)⇒(trajectory_size,)にするため。calc_stdev.update_mean_var()はそう想定。
            _, _, sigma_reward_accumulated = self._calc_stdev.update_mean_var(trajectory["reward"].squeeze())       
            
            #GAEの計算本体
            #同時にCriticの教師信号も計算
            gae, Vtarg = self._calculate_GAE_and_Vtarg(trajectory, sigma_reward_accumulated, lamda_GAE, gamma)
            #以下、gaeとVtagはndarrayであることに注意　shapeは(trajectory_size, 1)のはず
            trajectory["GAE"] = gae
            trajectory["Vtarg"] = Vtarg
            
            ##Actorの訓練##

            loss_actor = self._train_actor(trajectory, gamma, batch_size, clip_range)
            
            ##Criticの訓練##

            loss_critic = self._train_critic(trajectory, batch_size)

            ##訓練成果記録のためのPlay　1エピソード　step数とscoreを記録する##
            
            #訓練成果記録のためのPlayのenvは、訓練時の独自報酬設計のwrapperではなく、wrapされているオリジナルのenv
            env_wrapped = self._env.env
            
            done = False
            total_steps = 0
            total_reward = 0
            save_temp_params = False
            st = env_wrapped.reset()
            while (done==False):
                
                #agentに最適行動を推測させる
                #最適行動と、その最適行動の正規分布上での確率密度関数値が返される
                a, _ = self._agent.actor.predict_best_action_and_policy(st)
                #最適行動で1step進める
                next_st, rew, done, _  = env_wrapped.step(a)
                
                total_steps += 1
                total_reward += rew
                
                st = next_st 
                
            if total_reward>=best_score:
                #scoreで成績を計測
                best_score = total_reward
                best_score_count += 1
                save_temp_params = True
                
            if save_temp_params==True:
                #訓練対象パラメーターを一時退避
                self._agent.keep_temporarily_learnable_params()
                
            #このエポックでのlossや訓練成果の記録
            loss_actor_epochs.append(loss_actor)
            loss_critic_epochs.append(loss_critic)
            steps_epochs.append(total_steps)
            score_epochs.append(total_reward)

            if verbose_interval>0 and ( (epc+1)%verbose_interval==0 or epc==0 or (epc+1)==epochs ):
                summary_epc = "Epoch:" + str(epc) + " score:" + str(total_reward) + " steps:" + str(total_steps) + " loss_actor:" + str(loss_actor) + " loss_critic:" + str(loss_critic)
                summary_epc = summary_epc + " best score:" + str(best_score) + "(" + str(best_score_count) + "回)"
                if save_temp_params==True:
                    summary_epc = summary_epc + " パラメーター一時退避"
                time_string = datetime.now().strftime('%H:%M:%S')
                summary_epc = summary_epc + " time:" + time_string
                print(summary_epc)
                
        #一時退避させてたパラメーターを戻す
        self._agent.adopt_learnable_params_kept_temporarily()
        if verbose_interval>0:
            print("一時退避したパラメーターを正式採用")
        
        end_time = datetime.now()        
        
        processing_time_total = end_time - start_time #総処理時間　datetime.timedelta
        processing_time_total_string = timedelta_HMS_string(processing_time_total) #総処理時間の文字列表現
        
        if verbose_interval>0:
            print("総処理時間：", processing_time_total_string)
            
        result = {}
        result["name"] = self._name
        result["loss_actor_epochs"] = loss_actor_epochs
        result["loss_critic_epochs"] = loss_critic_epochs
        result["steps_epochs"] = steps_epochs
        result["score_epochs"] = score_epochs
        result["best_score"] = best_score
        result["processing_time_total_string"] = processing_time_total_string
        result["processing_time_total"] = processing_time_total
        #以下引数
        result["epochs"] = epochs
        result["trajectory_size"] = trajectory_size
        result["lamda_GAE"] = lamda_GAE
        result["gamma"] = gamma
        result["batch_size"] = batch_size
        result["clip_range"] = clip_range
        result["standardize_rewards_per_epoch"] = standardize_rewards_per_epoch
        
        return result


    def _calculate_GAE_and_Vtarg(self, trajectory, sigma_reward_accumulated, lamda_GAE, gamma):

        Vs = self._agent.critic.predict_V(trajectory["state"])
        next_Vs = self._agent.critic.predict_V(trajectory["next_state"])
        
        rewards_scaled = trajectory["reward"] / (sigma_reward_accumulated + 1e-4)
        deltas = rewards_scaled + gamma * (1 - trajectory["done"]) * next_Vs - Vs
        
        ##GAE　計算##

        gaes = np.zeros_like(deltas, dtype=np.float32)        
        gae = 0
        for i in reversed( range( len(deltas) ) ):
            gae = deltas[i] + gamma * lamda_GAE * (1 - trajectory["done"][i]) * gae
            gaes[i] = gae

        ##Criticの教師信号　計算##

        #教師信号 = Advantage + 出力値V
        Vtargs = gaes + Vs

        return gaes.reshape(-1, 1), Vtargs.reshape(-1, 1)
    
    def _train_actor(self, trajectory, gamma, batch_size, clip_range):

        xsize = trajectory["state"].shape[0]
        idx = np.arange(xsize)
        np.random.shuffle(idx)
        iter_num = np.ceil(xsize / batch_size).astype(np.int) #イテレーション数
        
        losses_ = []

        for it in range(iter_num):
            
            mask = idx[batch_size*it : batch_size*(it+1)]
            
            #ミニバッチの生成
            states_ = tf.convert_to_tensor(trajectory["state"][mask])
            actions_ = tf.convert_to_tensor(trajectory["action"][mask])
            gaes_ = tf.convert_to_tensor(trajectory["GAE"][mask])
            policies_ = tf.convert_to_tensor(trajectory["policy"][mask])

            #ミニバッチをActorに渡して訓練 train_on_batch
            loss_ = self._agent.actor.train(states_, actions_, gaes_, policies_, clip_range)
            
            losses_.append(loss_)

        loss_mean = np.mean(losses_)

        return loss_mean

    def _train_critic(self, trajectory, batch_size):

        xsize = trajectory["state"].shape[0]
        idx = np.arange(xsize)
        np.random.shuffle(idx)
        iter_num = np.ceil(xsize / batch_size).astype(np.int) #イテレーション数
        
        losses_ = []

        for it in range(iter_num):
            
            
            mask = idx[batch_size*it : batch_size*(it+1)]
            
            #ミニバッチの生成
            states_ = tf.convert_to_tensor(trajectory["state"][mask])
            Vtargs_ = tf.convert_to_tensor(trajectory["Vtarg"][mask])

            #ミニバッチをCriticに渡して訓練 train_on_batch
            loss_ = self._agent.critic.train(states_, Vtargs_)
            
            losses_.append(loss_)

        loss_mean = np.mean(losses_)

        return loss_mean          

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent
        
    class Calculater_Statistics:
        
        #ある分布の平均と分散を逐次計算
        #対象統計データ量が多くなる場合に使用（多くなければ本クラスを使用せず、例えば標準偏差ならnp.stdでよい）
        #よって、対象統計データそのものを保持せず、その平均と分散のみを保持する。
        #追加データを加味した平均と分散を追加都度計算し直し、それらを返す。
        
        id_instance = 0
        
        def __init__(self, id, shape=None, dtype=np.float32, description=""):
            
            self._mean_curr = np.zeros(shape, dtype) #現時点での対象統計データの平均
            self._var_curr = np.zeros(shape, dtype) #現時点での対象統計データの分散
            self._count_curr = 0 #現時点での対象統計データの件数
            
            self._shape = shape
            self._dtype = dtype
            
            self._id = id #インスタンスが複数作成される場合の区別に使用          
            self._description = description #このインスタンスの使用目的に使用　「to calculate the stdev of the accumulated rewards」など
        
        @classmethod
        def createInstance(cls, shape=None, dtype=np.float32, description=""):
            
            id_ins = cls.id_instance
            ins = cls(id_ins, shape, dtype, description)  
            
            cls.id_instance += 1
            
            return ins
        
        def update_mean_var(self, X_added):
            
            #X_added：追加するデータ　　shapeは(追加データ数, shape)
            #ただし、shapeがNoneや()の場合、shapeは(追加データ数, )
            #　使用される場面は、train()中あるエポックでのtrajectory収集後の蓄積rewardの標準偏差算出
            #　X_addedは、そのエポックでの全stepのrewardsのndarray　shapeは(このエポックでのstep数, 1)
            
            mean_added = np.mean(X_added, axis=0)
            var_added = np.var(X_added, axis=0)
            count_added = X_added.shape[0]
            
            mean_delta = mean_added - self._mean_curr
            var_added_multiplied_count = var_added * count_added
            var_curr_multiplied_count = self._var_curr * self._count_curr
            
            count_new = self._count_curr + count_added
            mean_new =self._mean_curr + ( mean_delta * count_added / count_new )            
            var_new = (  var_curr_multiplied_count + var_added_multiplied_count \
                      + ( np.square(mean_delta) * self._count_curr * count_added / count_new )  ) \
                      / count_new
            
            self._mean_curr = mean_new
            self._var_curr = var_new
            self._count_curr = count_new
            
            return mean_new, var_new, np.sqrt(var_new)                         
            
        
        @property
        def id(self):
            return self._id
        
        @property
        def shape(self):
            return self._shape
        
        @property
        def dtype(self):
            return self._dtype
        
        @property
        def description(self):
            return self._description
        
        @property
        def curr_mean(self):
            return self._mean_curr
        
        @property
        def curr_var(self):
            return self._curr_var
        
        @property
        def curr_stdev(self):
            return np.sqrt(self._curr_var)