# -*- coding: utf-8 -*-
#Agent
#ActorとCriticの2つのNNを内包する
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np
import os

tfd = tfp.distributions

class Agent():
    
    def __init__(self, name, state_dim, action_dim):
        
        self._name = name
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        #Actorインスタンス生成
        self._actor = Agent.Actor(state_dim, action_dim)
        
        #Criticインスタンス生成
        self._critic = Agent.Critic(state_dim, action_dim)
        
    def keep_temporarily_learnable_params(self):
        
        self._actor.keep_temporarily_learnable_params()
        self._critic.keep_temporarily_learnable_params()
        
    def adopt_learnable_params_kept_temporarily(self):
        
        self._actor.adopt_learnable_params_kept_temporarily()
        self._critic.adopt_learnable_params_kept_temporarily()        
        
    def save_learnable_params(self, parent_dir, files_dir_name="", common_file_name=""):
        
        #parent_dir/files_dir_name/の下に、common_file_name+"_actor" 、common_file_name+"_critic" 
        #の2つ（以上）のlearnable_paramsファイルを作成
        
        if files_dir_name=="":
            files_dir_name = self._name
            
        if common_file_name=="":
            common_file_name = self._name
            
        files_dir = os.path.join(parent_dir, files_dir_name)
        
        os.makedirs(files_dir, exist_ok=True)
        
        path_actor = os.path.join(files_dir, common_file_name + "_actor")
        path_critic = os.path.join(files_dir, common_file_name + "_critic")        
        
        self._actor.save_weights(path_actor)
        self._critic.save_weights(path_critic)

        return files_dir, common_file_name
        
    def read_learnable_params(self, files_dir, common_file_name):
        
        #files_dir/の下の、common_file_name+"_actor" 、common_file_name+"_critic" 
        #の2つ（以上）のlearnable_paramsファイルの中身をload
        
        #注意！！この機能では、「訓練の継続」はできない。訓練済モデルによる「推論」のみである。
        #訓練途中の累積報酬の平均と分散を保持・計算するCalculater_Statisticsは、GAEを計算するために必須のものである。
        #だがそれは1個上の階層のTrainerが保持しているので、このAgentでは保存できずしていない。
        #なのでこのreadで復活するパラメーターは訓練にとっては不完全。
        #Calculater_Statisticsは、推論時には使用しないので、推論は問題なく可能。
        
        path_actor = os.path.join(files_dir, common_file_name + "_actor")
        path_critic = os.path.join(files_dir, common_file_name + "_critic")
        
        self._actor.load_weights(path_actor)
        self._critic.load_weights(path_critic)

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def state_dim(self):
        return self._state_dim
    
    @property
    def action_dim(self):
        return self._action_dim
    
    #with tf.GradientTape()内の順伝播を考えると、やはり直接ActorとCriticの順伝播をcallできた方が良いのでは。
    #そのために、内部のインスタンスであるactorとcriticをgetterプロパティとしてpublicインターフェースとする。
    
    @property
    def actor(self):
        return self._actor
    
    @property
    def critic(self):
        return self._critic

        
    class Actor(tf.keras.Model):
        
        def __init__(self, state_dim, action_dim):
            
            super(Agent.Actor, self).__init__()
            
            self._state_dim = state_dim
            self._action_dim = action_dim
            
            self._define_layers()
            
            self._optimizer = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            
            #訓練対象パラメーターの一時退避先。layerごとのdictionary。
            #https://note.nkmk.me/python-tensorflow-keras-get-layer-index/
            #https://www.tensorflow.org/tutorials/customization/custom_layers?hl=ja
            self._temp_weights = {}
            self._temp_biases = {}

            #いったんダミーデータで順伝播し、計算グラフを構築する。
            #summay()出力したい場合と、計算グラフ未構築時点でsave_weightsされる可能性がありエラーなく対処したい場合にのみ行うべき。
            dummy_st = np.zeros((1, state_dim), dtype=np.float32)
            dummy_st = tf.convert_to_tensor(dummy_st)
            out = self(dummy_st)
            #self.summary()
                    
        def _define_layers(self):

            output_dim_afn1 = self._state_dim*10
            output_dim_afn3 = self._action_dim*10
            output_dim_afn2 = np.ceil( np.sqrt(output_dim_afn1*output_dim_afn3) ).astype(np.int)
            
            self._afn1 = kl.Dense(name="a_afn1", units=output_dim_afn1, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            self._afn2 = kl.Dense(name="a_afn2", units=output_dim_afn2, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            self._afn3 = kl.Dense(name="a_afn3", units=output_dim_afn3, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            #平均μの出力のlayer
            self._afn_mu = kl.Dense(name="a_afn_mu", units=self._action_dim, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            #標準偏差σの出力のlayer
            self._afn_sigma = kl.Dense(name="a_afn_sigma", units=self._action_dim, activation="softplus", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')

        #以下の警告が出る。     
        #WARNING:tensorflow:5 out of the last 5 calls to <function Agent.Actor.call at 0x7fae2bbf1400> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
        #https://www.tensorflow.org/tutorials/customization/performance?hl=ja
        #の「引数は Python か？ Tensor か？」を読んで対処する。
        #呼び出し時、tensorflowのtensorにcastするのは必須。
        #@tf.function関連では、以下
        #https://qiita.com/t_shimmura/items/1209d01f1e488c947cab
        #の、「Tracing」の節が非常に参考になる。
        #また、https://blog.shikoan.com/tf-function-tracing/
        #も、種々の計測をしており、参考になる。
        @tf.function
        def call(self, state):
            #NNの順伝播　平均と標準偏差を返す
            #tf仕様上は必須ではないが、入力値stateをtensorにして呼ぶこと

            x = self._afn1(state)
            x = self._afn2(x)
            x = self._afn3(x)

            mu = self._afn_mu(x)
            sigma = self._afn_sigma(x)

            return mu, sigma
        
        def keep_temporarily_learnable_params(self):
            #訓練対象パラメーター一時退避
            
            #self._temp_learnable_params = self.trainable_variables　はダメ。
            #trainable_variablesはgetterで、それでgetしたものをadoptするとき困る。
            for l in self.layers:
                self._temp_weights[l.name] = l.kernel
                self._temp_biases[l.name] = l.bias
            
        def adopt_learnable_params_kept_temporarily(self):
            #一時退避した訓練対象パラメーターを正式採用

            #self.trainable_variables = self._temp_learnable_params　はダメ。trainable_variablesはgetter。
            for l in self.layers:
                l.kernel = self._temp_weights[l.name]
                l.bias = self._temp_biases[l.name] 

        def predict_mean_and_stdev(self, states):
            #NNの順伝播出力　平均と標準偏差を返す
            #ただし入出力値がndarrayであることが、call()と異なる

            #call()の入力のため、入力値をtensorに
            states = np.atleast_2d(states).astype(np.float32)
            states = tf.convert_to_tensor(states)

            #call()を呼ぶ
            mu, sigma = self(states)

            #出力値をndarrayに
            mu = mu.numpy().reshape(-1, self._action_dim)
            sigma = sigma.numpy().reshape(-1, self._action_dim)

            return mu, sigma
        
        def predict_best_action_and_policy(self, a_state):
            #最適行動を提示
            #その最適行動の確率密度関数値を一緒に返す
            
            #NNが順伝播入力として受け付けるshapeは、（N, state_dim）の2dのみ。
            a_state = np.atleast_2d(a_state).astype(np.float32)
            
            if a_state.shape[0]!=1:
                raise ValueError("a_stateは1件だけにしてください。1件のstateについてbest actionを推測します。") 
            
            #call()に投げる引数は必ずtfのTensorにすること
            a_state = tf.convert_to_tensor(a_state)
            
            #ActorのNNの順伝播
            mu, sigma = self(a_state)
            
            #NNの純粋な出力値であるmuとsigmaのshapeは必ず(1, action_dim)になっている。
            #そのshapeのaxis=0（「1」）は、以下の正規分布の確率密度関数値の算出では冗長であり、ここで削除しておく。
            mu = tf.squeeze(mu)
            sigma = tf.squeeze(sigma)
            
            #この時点で、muとsigmaのshapeは(action_dim,)のはず
            
            #平均mu、標準偏差sigmaの正規分布の確率密度関数　
            #muやsigmaのshapeは(action_dim,)なので、関数も同じくaction_dim個ある。
            pdf_normal = tfd.Normal(loc=mu, scale=sigma)
            
            #正規分布の確率密度関数からactionをサンプリング
            #shapeは(action_dim,)
            best_action = pdf_normal.sample()
            
            #サンプリングしたaction（横軸）に相当する確率密度関数値（縦軸）
            #shapeは(action_dim,)
            prob = pdf_normal.prob(best_action)

            #numpy配列、かつ最低でもベクトル（1次元）で返す
            #戻り値であるbest_actionは、env.step()の引数にこのまま使用されるため。
            #shapeが(action_dim, )でないと（例えば(action_dim, 1)とかスカラーとかだと）、env.step()の戻り値のshapeがおかしくなったり、特にスカラーの場合はエラーになったりする。
            #Pendulumはaction_dim=1で、知らない間にスカラーになってそのままenv.step()に渡すと、エラーとなる。それを防止するためのnp.atleast_1d。
            best_action = np.atleast_1d(best_action.numpy())
            prob = np.atleast_1d(prob.numpy())
            
            return best_action, prob

        @tf.function
        def train(self, states, actions, gaes, base_policies, clip_range, loss_actor_entropy_coef):
            #Actorの訓練関数
            #引数の訓練データはミニバッチ
            #tf仕様上は必須ではないが、引数の訓練データはtensorにして呼ぶこと
            
            #print("states.shape:", states.shape) #(batch_size, state_dim)
            #print("actions.shape:", actions.shape) #(batch_size, action_dim)
            #print("gaes.shape:", gaes.shape) #(batch_size, 1)
            #print("base_policies.shape:", base_policies.shape) #(batch_size, action_dim)
            
            #経験データbase_policiesは、経験データactionsのもとになった正規分布の確率密度関数の関数値（縦軸）で、
            #actionsと同じくaction_dim毎、つまりshapeは(batch_size, action_dim)
            #logを取って、全action_dimで合計
            log_base_policies = tf.math.log(base_policies + 1e-8)
            log_base_policies_sum = tf.reduce_sum(log_base_policies, axis=1, keepdims=True)
            #(batch_size, 1)

            with tf.GradientTape() as tape:
                
                #Actorの順伝播
                #正規分布の平均と標準偏差を出力させる
                curr_mus, curr_sigmas = self(states)
                #双方、(batch_size, action_dim)

                #Actorの順伝播で得られた平均と標準偏差をもとに、正規分布の確率密度関数を生成
                pdfs_normal = tfd.Normal(loc=curr_mus, scale=curr_sigmas)
                
                #経験データactions（横軸）に対応する確率密度関数値（縦軸）を算出
                curr_policies = pdfs_normal.prob(actions)
                #(batch_size, action_dim)
                
                #logを取って、全action_dimで合計
                log_curr_policies = tf.math.log(curr_policies + 1e-8)
                log_curr_policies_sum = tf.reduce_sum(log_curr_policies, axis=1, keepdims=True)
                #(batch_size, 1)
                
                #ratiosは、NNパラメーター更新幅の抑制に使用される。
                #よって、action_dim毎に持たない。
                #そのために、log_curr_policiesとlog_base_policiesをaxis=1でsumし、各々を(batch_size, 1)にした。
                #ratios = curr_policiesをaxis=1でsumしたもの) / base_policiesをaxis=1でsumしたもの
                ratios = tf.math.exp(log_curr_policies_sum - log_base_policies_sum)
                #(batch_size, 1)
                
                #clipしたratio
                ratios_clipped = tf.clip_by_value(ratios, 1-clip_range, 1+clip_range)
                #(batch_size, 1)

                #clipしない目的関数値
                values_obj_func_unclipped = ratios * gaes
                #(batch_size, 1)

                #clipした目的関数値
                values_obj_func_clipped = ratios_clipped * gaes
                #(batch_size, 1)

                #目的関数値　上記両者の大きくない方
                values_obj_func = tf.minimum(values_obj_func_unclipped, values_obj_func_clipped)
                #(batch_size, 1)
                
                #最小化すべき損失関数値　目的関数値をただマイナスする
                losses = -1 * values_obj_func
                #(batch_size, 1)
                
                #損失関数値のエントロピー補正項
                if loss_actor_entropy_coef>0:
                    entropies_curr_policies = tf.reduce_sum(-1 * curr_policies * log_curr_policies, axis=1, keepdims=True)
                    #(batch_size, 1)
                    #curr_policiesの分布が”確定的”なほどentropyは小さい　→　しかし損失関数値は増やしたい（損失関数値の減少幅を小さく）
                    #curr_policiesの分布が”あいまい”なほどentropyは大きい　→　しかし損失関数値は減らしたい（損失関数値の減少幅を大きく）
                    #⇒本来の損失関数値からeentropyをマイナスする
                    #curr_policiesの分布が”確定的”なほどentropyは小さい　→　損失関数値の減少幅は小さくなる
                    #curr_policiesの分布が”あいまい”なほどentropyは大きい　→　損失関数値は減少幅は大きくなる
                    #注意！　-1 * curr_policies_sum * log_curr_policies_sum　としないように。数学的に等しくない。
                    #K=2として書いてみればわかる。
                    #正しい方(マイナスは省く)　： p1*logp1 + p2*logp2
                    #間違いの方（マイナスは省く）　：　(p1+p2) * log(p1+p2) = p1*log(p1+p2) + p2*log(p1+p2)　となり、↑と異なる。
                    losses = losses - loss_actor_entropy_coef * entropies_curr_policies
                
                loss = tf.reduce_mean(losses)

            grads = tape.gradient(loss, self.trainable_variables)
            
            self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

            return loss
            
    
    class Critic(tf.keras.Model):
            
        def __init__(self, state_dim, action_dim):
            
            super(Agent.Critic, self).__init__()
            
            self._state_dim = state_dim
            self._action_dim = action_dim
            
            self._define_layers()
            
            self._optimizer = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            
            #訓練対象パラメーターの一時退避先。layerごとのdictionary。
            #https://note.nkmk.me/python-tensorflow-keras-get-layer-index/
            #https://www.tensorflow.org/tutorials/customization/custom_layers?hl=ja
            self._temp_weights = {}
            self._temp_biases = {}

            #いったんダミーデータで順伝播し、計算グラフを構築する。
            #summay()出力したい場合と、計算グラフ未構築時点でsave_weightsされる可能性がありエラーなく対処したい場合にのみ行うべき。
            dummy_st = np.zeros((1, state_dim), dtype=np.float32)
            dummy_st = tf.convert_to_tensor(dummy_st)
            out = self(dummy_st)
            #self.summary()
            
        def _define_layers(self):

            output_dim_afn1 = self._state_dim*10
            output_dim_afn3 = self._action_dim*10
            output_dim_afn2 = np.ceil( np.sqrt(output_dim_afn1*output_dim_afn3) ).astype(np.int)
            
            self._afn1 = kl.Dense(name="c_afn1", units=output_dim_afn1, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            self._afn2 = kl.Dense(name="c_afn2", units=output_dim_afn2, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            self._afn3 = kl.Dense(name="c_afn3", units=output_dim_afn3, activation="tanh", use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            #価値関数Vの出力のlayer
            self._afn_V = kl.Dense(name="c_afn_V", units=1, use_bias=True, kernel_initializer='Orthogonal', bias_initializer='zeros')
            
        @tf.function
        def call(self, state):
            #NNの順伝播　価値関数を返す
            #tf仕様上は必須ではないが、入力値stateをtensorにして呼ぶこと

            x = self._afn1(state)
            x = self._afn2(x)
            x = self._afn3(x)

            V = self._afn_V(x)
            
            return V
        
        def keep_temporarily_learnable_params(self):
            #訓練対象パラメーター一時退避

            #self._temp_learnable_params = self.trainable_variables　はダメ。
            #trainable_variablesはgetterで、それでgetしたものをadoptするとき困る。
            for l in self.layers:
                self._temp_weights[l.name] = l.kernel
                self._temp_biases[l.name] = l.bias
            
        def adopt_learnable_params_kept_temporarily(self):
            #一時退避した訓練対象パラメーターを正式採用

            #self.trainable_variables = self._temp_learnable_params　はダメ。trainable_variablesはgetter。
            for l in self.layers:
                l.kernel = self._temp_weights[l.name]
                l.bias = self._temp_biases[l.name] 
        
        def predict_V(self, states):
            #NNの順伝播出力　価値関数を返す
            #ただし入出力値がndarrayであることが、call()と異なる

            #call()の入力のため、入力値をtensorに
            states = np.atleast_2d(states).astype(np.float32)
            states = tf.convert_to_tensor(states)

            #call()を呼ぶ
            V = self(states)

            #出力値をndarrayに
            V = V.numpy().reshape(-1, 1)

            return V

        @tf.function
        def train(self, states, Vtargs):
            #Criticの訓練関数
            #引数の訓練データはミニバッチ
            #tf仕様上は必須ではないが、引数の訓練データはtensorにして呼ぶこと

            with tf.GradientTape() as tape:

                Vs = self(states)
                losses = tf.square(Vs - Vtargs)
                loss = tf.reduce_mean(losses)

            grads = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

            return loss