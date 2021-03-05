# -*- coding: utf-8 -*-
#Trainer
#環境envとAgentを持ち、Agentの訓練を受け持つ
#PPOに沿ってAgentをtrainする

import tensorflow as tf
import numpy as np
from datetime import datetime

from common.funcs import *


class PPOAgentTrainer():
    
    def __init__(self, name, env, agent, description=""):
        
        self._VERSION = 2.0        
        
        self._name = name
        self._env = env
        self._agent = agent

        self._state_dim = self._agent.state_dim
        self._action_dim = self._agent.action_dim
        
        self._description = description
                       
        #train後のオブジェクトインスタンスに対して追加でtrain()することがよくある。
        #その際、max_total_rewを引き継がないで毎回max_total_rewを-np.infで初期化してしまうと、
        #せっかく1回目のtrainで良いtotal_rewを出してその時のパラメーターを正式採用しても、
        #2回目のtrainの1エポック目でmax_total_rew判定され、問答無用にその1エポック目の更新後パラメーターを一時退避してしまう。
        #つまり1回目の良いパラメーターは捨てられてしまう。
        #それを避けるためには、max_total_rewを引き継ぐ必要がある。
        self._max_total_rew = -np.inf #train()中の稼得即時報酬合計のそれまでの最大値
        self._max_total_rew_count = 0 #train()中の稼得即時報酬合計のそれまでの最大値　更新回数        
        #現時点での訓練対象パラメーターを一時退避　train()にてmax_total_rew達成によるパラメーター一時退避が起こらないこともありうる
        self._agent.keep_temporarily_learnable_params()
        
        #報酬の蓄積
        #即時報酬標準化のための全報酬の標準偏差の算出に使用
        #経験バッファと異なり、エポックで洗い替えない（train開始時点から蓄積し続ける）
        #上記max_total_rew関連同様、train()を連続して呼ばれることを考慮し、メンバ変数とする
        self._calc_stdev = Calculater_Statistics.createInstance()
        
        
    def train(self, epochs, trajectory_size=1024, lamda_GAE=0.95, gamma=0.99, batch_size=1024, clip_range=0.2, 
              loss_actor_entropy_coef=0, verbose_interval=1):
        
        #epochs：何エポック訓練するか
        #trajectory_size：経験データのサイズ
        #lamda_GAE：GAE算出の際に使用するλ
        #gamma：報酬の割引率γ
        #batch_size：1イテレーションのミニバッチサイズ
        #clip_range：ratio算出の際に使用するclipの範囲のε
        #loss_actor_entropy_coef：actor（policy側）のlossにpolicyのentropyを適用する係数（０以上）
        #verbose_interval：何エポック毎に訓練記録をprint出力するか
        
        start_time = datetime.now()
        
        #訓練結果の記録　エポック毎の記録
        loss_actor_epochs = [] #Actorのlossのエポック毎の記録　正確には、そのエポックでのNN更新時の各イテレーションでの順伝播lossのイテレーション平均
        loss_critic_epochs = [] #Criticのlossのエポック毎の記録　正確には、そのエポックでのNN更新時の各イテレーションでの順伝播lossのイテレーション平均
        total_reward_epochs = [] #経験データ収集時の即時報酬合計のエポック毎の記録
        total_reward_orig_epochs = [] #経験データ収集時の即時報酬（オリジナル）合計のエポック毎の記録（訓練成果報告用）
        avg_steps_epochs = [] #経験データ収集時のステップ数エピソード平均のエポック毎の記録
        avg_reward_epochs = [] #経験データ収集時の即時報酬エピソード平均のエポック毎の記録
        avg_reward_orig_epochs = [] #経験データ収集時の即時報酬（オリジナル）エピソード平均のエポック毎の記録（訓練成果報告用）
        
        #（重要）各エポックでの稼得即時報酬合計total_rewは、metricsとなる
        
        #rew_origの「_orig」は、独自報酬設計を適用されていない、wrapされたenvが返す即時報酬及びそれをもとにした数値。
        #訓練成果「報告」用。世間一般のweb上、書籍上の数値はみなこのオリジナルの報酬値であり、横比較するため「だけ」にこの「_ori」はある。
        #「_ori」が無いのは独自報酬設計を適用した数値であり、訓練成果の判定に使用される。verbose=1で画面に表示されるのはこちら。
        #「avg_」はエピソード平均。
        #trajectoryサイズに達して経験データ収集を終了したときは、ほぼ確実に最終エピソードの途中である。
        #その途中エピソードもエピソード数「1」とする。 
        
        trajectory = None
        
        for epc in range(epochs):
            
            #1エポック
            #（初回エポックのみ）経験データ（trajectory）収集
            #収集済経験データについて、GAEを算出
            #収集済み経験データとGAEで訓練　Agent（ActorとCritic）を更新
            #このエポックでの訓練成果の評価と、次のエポックでの訓練のための経験データ準備の両方を兼ねて、経験データ（trajectory）収集
            #収集した経験データを評価、ステップ数、即時報酬、即時報酬（オリジナル）のエピソード平均など
            #訓練評価結果によっては、訓練対象パラメーターを一時退避
            #訓練評価結果の表示
            
            #このエポックで、訓練対象パラメーターを一時退避するかしないか
            save_temp_params = False
            
            ##初回エポックのみ、ここで経験データ（trajectory）収集##
            if epc==0 or trajectory is None:                
                
                #実際にAgentを走らせて、経験データ（trajectory）生成
                trajectory, episode_count = self._create_trajectory(trajectory_size)
            
            ##GAEの算出##

            #即時報酬の標準化（Reward Scaling）のために、訓練当初より蓄積された即時報酬の標準偏差を求める。
            #後のGAE算出時、訓練当初より蓄積された即時報酬の標準偏差で各即時報酬を割る。
            #ただし平均は引かない。
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

            loss_actor = self._train_actor(trajectory, gamma, batch_size, clip_range, loss_actor_entropy_coef)
            
            ##Criticの訓練##

            loss_critic = self._train_critic(trajectory, batch_size)

            ##このエポックでの訓練成果の評価と、次のエポックでの訓練のための経験データ準備の両方を兼ねて、経験データ（trajectory）収集##
            
            #実際にAgentを走らせて、経験データ（trajectory）生成
            trajectory, episode_count = self._create_trajectory(trajectory_size)
            
            ##収集した経験データを評価、ステップ数、即時報酬、即時報酬（オリジナル）のエピソード平均など　必要ならば訓練対象パラメーターを一時退避##
            
            #即時報酬の合計、即時報酬やステップ数のエピソード平均の算出
            total_rew, total_rew_orig, avg_steps, avg_rew, avg_rew_orig = self._calc_total_avg_trajectory(episode_count, trajectory)
            
            #metricsである即時報酬の合計total_rewが、その今までの最大値以上なら、現時点でのパラメーターを一時退避
            if total_rew>=self._max_total_rew:
                self._max_total_rew = total_rew
                self._max_total_rew_count += 1
                #訓練対象パラメーターを一時退避
                save_temp_params = True
                self._agent.keep_temporarily_learnable_params()
            else:
                save_temp_params = False
            
            #このエポックでのlossや訓練成果の記録
            loss_actor_epochs.append(loss_actor)
            loss_critic_epochs.append(loss_critic)            
            total_reward_epochs.append(total_rew)
            total_reward_orig_epochs.append(total_rew_orig)
            avg_steps_epochs.append(avg_steps)
            avg_reward_epochs.append(avg_rew)
            avg_reward_orig_epochs.append(avg_rew_orig)
            
            #画面表示のための文字列生成
            if verbose_interval>0 and ( (epc+1)%verbose_interval==0 or epc==0 or (epc+1)==epochs ):
                summary_epc = "Epoch:" + str(epc) + " total_rew:" + str(round(total_rew, 2))
                summary_epc = summary_epc + " Avg[rew:" + str(round(avg_rew)) + " step:" + str(round(avg_steps))  + "] loss_actor:" + str(round(loss_actor, 2)) + " loss_critic:" + str(round(loss_critic, 2))
                summary_epc = summary_epc + " max_total_rew:" + str(round(self._max_total_rew, 2)) + "(" + str(self._max_total_rew_count) + "回)"
                if save_temp_params==True:
                    summary_epc = summary_epc + " params一時退避"
                time_string = datetime.now().strftime('%H:%M:%S')
                summary_epc = summary_epc + " time:" + time_string
                print(summary_epc)
        
        #エポック反復終了
        
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
        result["name"] = self._name #このインスタンスの名前
        result["description"] = self._description #このインスタンスの説明
        result["loss_actor_epochs"] = loss_actor_epochs #各エポックでのActorのlossのList。Listの1要素はエポック。
        result["loss_critic_epochs"] = loss_critic_epochs #各エポックでのCriticのlossのList。Listの1要素はエポック。
        result["total_reward_epochs"] = total_reward_epochs #各エポックでの即時報酬エピソード平均のList。Listの1要素はエポック。
        result["total_reward_orig_epochs"] = total_reward_orig_epochs #各エポックでの即時報酬エピソード平均のList。Listの1要素はエポック。
        result["avg_steps_epochs"] = avg_steps_epochs #各エポックでのステップ数エピソード平均のList。Listの1要素はエポック。
        result["avg_reward_epochs"] = avg_reward_epochs #各エポックでの即時報酬エピソード平均のList。Listの1要素はエポック。
        result["avg_reward_orig_epochs"] = avg_reward_orig_epochs #各エポックでの即時報酬（オリジナル）エピソード平均のList。Listの1要素はエポック。
        result["max_total_reward"] = self._max_total_rew #各エポックでの即時報酬合計の最大値。
        result["processing_time_total_string"] = processing_time_total_string #総処理時間の文字列表現。
        result["processing_time_total"] = processing_time_total #総処理時間。
        #以下引数など
        result["VERSION"] = self._VERSION
        result["epochs"] = epochs
        result["trajectory_size"] = trajectory_size
        result["lamda_GAE"] = lamda_GAE
        result["gamma"] = gamma
        result["batch_size"] = batch_size
        result["clip_range"] = clip_range
        result["loss_actor_entropy_coef"] = loss_actor_entropy_coef
        
        return result


    def _create_trajectory(self, trajectory_size):
        
        #経験データTrajectoryを生成
        #Agentをtrajectory_sizeステップ分走らせる
        
        #env初期化
        st = self._env.reset()        
        
        #エピソード番号
        #必ず１から始める。エピソード数としても使用する。
        episode_num = 1
        #episode_nums = [] #現在のエピソード番号を入れる箱
        
        #trajectoryの容器
        #全データの特定の列に対して演算や置換を行うので、列の集合体とする
        trajectory = {"state": [], "action": [], "reward": [], "next_state": [], "done": [], "policy": [], 
                      "reward_orig": [], "reward_raw": [], "GAE":[], "Vtarg":[]}
            
        #経験バッファサイズまでstepし、経験データを経験バッファに保存
        for d in range(trajectory_size):
                
            #agentに最適行動を推測させる
            #最適行動と、その最適行動の正規分布上での確率密度関数値が返される
            a, pi = self._agent.actor.predict_best_action_and_policy(st)
            #最適行動で1step進める
            next_st, rew, done, _, rew_orig = self._env.step_ex(a)
            #rew_origは、wrapしているオリジナルのenvが返してきた（こちらの報酬設計を通っていないオリジナルの）即時報酬。訓練成果報告用数値。
                
            if done==True:
                next_st = self._env.reset()
                #エピソード終端時、trajectory["next_state"]に入るのは、時系列的に継続性のない、State初期値となる。
                #後のdelta計算時、エピソード最終ステップに対応するV(St+1)は（算出はされるが）使用されないので、↑で構わない。
                episode_num += 1
                
            #経験バッファに追加
            #episode_nums.append(episode_num)
            trajectory["state"].append(st)
            trajectory["action"].append(a)
            trajectory["reward_raw"].append(rew)
            trajectory["reward"].append(rew)
            trajectory["reward_orig"].append(rew_orig)
            trajectory["next_state"].append(next_st)
            trajectory["done"].append(done)
            trajectory["policy"].append(pi)
                
            #報酬の蓄積
            #reward_accumulated.append(rew)

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
        #episode_nums = np.array(episode_nums, dtype=np.int).reshape(-1, 1)
        trajectory["state"] = np.array(trajectory["state"], dtype=np.float32).reshape(-1, self._state_dim)
        trajectory["action"] = np.array(trajectory["action"], dtype=np.float32).reshape(-1, self._action_dim)
        trajectory["reward_raw"] = np.array(trajectory["reward_raw"], dtype=np.float32).reshape(-1, 1)
        trajectory["reward"] = np.array(trajectory["reward"], dtype=np.float32).reshape(-1, 1)
        trajectory["reward_orig"] = np.array(trajectory["reward_orig"], dtype=np.float32).reshape(-1, 1)
        trajectory["next_state"] = np.array(trajectory["next_state"], dtype=np.float32).reshape(-1, self._state_dim)
        trajectory["done"] = np.array(trajectory["done"], dtype=np.float32).reshape(-1, 1)
        trajectory["policy"] = np.array(trajectory["policy"], dtype=np.float32).reshape(-1, self._action_dim)
        
        return trajectory, episode_num
    

    def _calc_total_avg_trajectory(self, episode_count, trajectory):
        
        #以下を返す。
        #即時報酬、即時報酬（オリジナル）　それぞれの合計
        #即時報酬、即時報酬（オリジナル）、ステップ数　それぞれのエピソード平均
        
        steps = trajectory["reward"].shape[0]
        avg_steps = round(steps / episode_count)
        
        total_rew = np.sum(trajectory["reward"])
        avg_rew = total_rew / episode_count
        
        total_rew_orig = np.sum(trajectory["reward_orig"])
        avg_rew_orig = total_rew_orig / episode_count
        
        return total_rew, total_rew_orig, avg_steps, avg_rew, avg_rew_orig
        

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
    
    
    def _train_actor(self, trajectory, gamma, batch_size, clip_range, loss_actor_entropy_coef):

        xsize = trajectory["state"].shape[0]
                
        #以下は復元抽出
        #しかし実験の結果、「非復元抽出時より確実に低い、ある一定のscoreで停滞」「成長途中で大崩れ」の2パターンになったので、不採用
        #iter_num = np.ceil(xsize / batch_size).astype(np.int) * 3 #イテレーション数
        #idx = np.random.choice( range(xsize), (iter_num, batch_size) )
        
        #以下は非復元抽出
        iter_num = np.ceil(xsize / batch_size).astype(np.int) #イテレーション数
        idx = np.arange(xsize)
        np.random.shuffle(idx)
                
        losses_ = []

        for it in range(iter_num):
            
            #以下は復元抽出
            #mask = idx[it]
            
            #以下は非復元抽出
            mask = idx[batch_size*it : batch_size*(it+1)]
            
            #ミニバッチの生成
            states_ = tf.convert_to_tensor(trajectory["state"][mask])
            actions_ = tf.convert_to_tensor(trajectory["action"][mask])
            gaes_ = tf.convert_to_tensor(trajectory["GAE"][mask])
            policies_ = tf.convert_to_tensor(trajectory["policy"][mask])

            #ミニバッチをActorに渡して訓練 train_on_batch
            loss_ = self._agent.actor.train(states_, actions_, gaes_, policies_, clip_range, loss_actor_entropy_coef)
            
            losses_.append(loss_)

        loss_mean = np.mean(losses_)

        return loss_mean

    def _train_critic(self, trajectory, batch_size):

        xsize = trajectory["state"].shape[0]
                
        #以下は復元抽出
        #しかし実験の結果、「非復元抽出時より確実に低い、ある一定のscoreで停滞」「成長途中で大崩れ」の2パターンになったので、不採用
        #iter_num = np.ceil(xsize / batch_size).astype(np.int) * 3 #イテレーション数
        #idx = np.random.choice( range(xsize), (iter_num, batch_size) )
        
        #以下は非復元抽出
        iter_num = np.ceil(xsize / batch_size).astype(np.int) #イテレーション数
        idx = np.arange(xsize)
        np.random.shuffle(idx)
                
        losses_ = []

        for it in range(iter_num):
            
            #以下は復元抽出
            #mask = idx[it]
            
            #以下は非復元抽出
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
    def VERSION(self):
        return self._VERSION
    
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
    
    @property
    def description(self):
        return self._description
    
    @name.setter
    def description(self, description):
        self._description = description