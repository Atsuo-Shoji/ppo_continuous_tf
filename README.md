# PPO（Proximal Policy Optimization）（TensorFlow2を使用）


## 概要

PPO（Proximal Policy Optimization）をTensorflow2で実装しました。<br>
**PPOでは普通に行われる並列化Agentによる経験データ（Trajectory）収集は行っていません。Agentは単体です。**<BR>
    
本モデルは、行動が連続値を取る環境を対象としています。<BR>
本稿では、OpenAI GymのMountainCarContinuous、BipedalWalkerを使用しています。<br>

**※理論の説明はしていません。他のリソースを参考にしてください。**<br>
**&nbsp;&nbsp; ネットや書籍でなかなか明示されておらず、私自身が実装に際し収集や理解に不便を感じたものを記載しています。**

<br>
    
### 使用した環境
本稿の実験においては、以下の環境を使用しています。<br>

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|MountainCarContinuous|![MountainCarContinuous_mini](https://user-images.githubusercontent.com/52105933/100495002-8154d380-318a-11eb-82ab-e890ee97e411.png)|2|1|999<br>固定で999ステップ/エピソード|車をback and forthさせて山の頂上へ|
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|遠くまで歩いて前進|

<br>
    
## 実装の要点

### 全体の構成

Trainerは、AgentをPPOメソッドに従い訓練します。<br>
Agentの中身は、Actor-Criticです（両者独立したNN）。<BR>
PPOでは普通に行われる並列化Agentによる経験データ収集は行っていません。Agentは単体です。

![全体構成図](https://user-images.githubusercontent.com/52105933/100536516-740c1780-3264-11eb-8493-813815df1881.png)

#### Agentの構成

Actor-Criticで、ActorとCriticは別々のNNにしています。<BR>

Actor側では、行動はガウス方策に従います。<br>
よって、Actorの出力は、正規分布の平均μと標準偏差σとなります。<br>
行動の次元数をKとすると、平均μと標準偏差σの1セットがK個ある、ということになります。<br>

Critic側は、単純に状態価値関数値を出力するだけです。

![NN構成図_70](https://user-images.githubusercontent.com/52105933/100530464-c71aa600-3235-11eb-9c98-2174015889b9.png)

<br>

### 訓練処理

#### 流れ

1エポック内で、以下のことを上から順に行います。<br>
- 経験データ（Trajectory）収集<br>
- Agentの訓練<br>
- 1エピソード試行（ここでの稼得scoreで訓練成果の計測をする意図）

これを複数エポック繰り返します。

![訓練処理の時間軸_70](https://user-images.githubusercontent.com/52105933/100698999-34b50680-33dd-11eb-8498-1fadd4debbe1.png)

#### ratio（「rt(Θ)」）

ActorのLossをclipするのに使用されるratioは、以下のように算出しています。

![ratio説明_80](https://user-images.githubusercontent.com/52105933/100543347-edbafa00-3292-11eb-9304-96e4e759762c.png)

#### 即時報酬の標準化

稼得した即時報酬の大小関係を維持しつつ・・
- 各環境（MountainCarやBipedalWalker）間での報酬の数値規模の違い<BR>
- 同一環境内での、訓練課程での報酬の数値規模の違い<BR>

を吸収するために、稼得した即時報酬の標準化を行っています。<BR>
この標準化された即時報酬を使用して、GAE（Generalized Advantage Estimation）の算出をします。<BR>
ただし、この標準化において、平均を引き算しません。<BR>
平均を引くと、実際の即時報酬とはプラスマイナスの符号が逆になってしまう標準化報酬が出てきてしまうからです。<BR>

また、標準化する対象範囲は、1エポックで収集する経験データのみの即時報酬ではなく（エポック毎に経験データを洗い替える）、訓練開始からの全経験データの即時報酬としています。<BR>
「同一環境内での、訓練課程での報酬の数値規模の違いの吸収」を考慮してこのようにしました。

<br><br>

## 結果

### MountainCarContinuous

#### この環境について

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|MountainCarContinuous|![MountainCarContinuous_mini](https://user-images.githubusercontent.com/52105933/100495002-8154d380-318a-11eb-82ab-e890ee97e411.png)|2|1|999<br>固定で999ステップ/エピソード|車をback and forthさせて山の頂上へ|

#### 訓練の設定値など

|項目|値など|
|      :---:     |      :---:      | 
|経験データサイズ|6144|
|バッチサイズ|2048|
|イテレーション回数/エポック|3|
|Clip Rangeのε|0.2|
|GAE算出でのλ|0.95|
|GAE算出での報酬の割引率γ|0.99|

#### ScoreとLoss

＜Score＞<br>
![MountainCarContinuous_score](https://user-images.githubusercontent.com/52105933/100699775-367fc980-33df-11eb-89ef-30cf02eb9198.png)

＜Loss＞<br>
![MountainCarContinuous_loss](https://user-images.githubusercontent.com/52105933/100699845-662ed180-33df-11eb-8418-5358c1b1d22b.png)

#### 考察

状態2次元、行動1次元という単純さのおかげなのか、訓練開始から一気にScoreを伸ばしました。<br>
両Lossもきれいに下がっていっています。<BR>
が、Scoreは0を漸近線として上昇が止まってしまいました。<BR>
この環境では、本来は、1エピソードでの即時報酬の合計（つまりScore）は60や70ということもありえます。<br>

並列化Agentによる経験データ収集を行っていないから、と推測します。<br>
異なったパラメーターによる「多様」で「大量」の経験データであれば、Scoreが60や70といった経験データもその中に含まれる可能性が出てきます。<br>
また、Agentが1個だけだと、収集できる経験データの個数から、イテレーション回数/エポックはせいぜい3～4回程度で、これでは更新回数が少ないと思います。<br>


### BipedalWalker

#### この環境について

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|遠くまで歩いて前進|

#### 訓練の設定値など

|項目|値など|
|      :---:     |      :---:      | 
|経験データサイズ|8192|
|バッチサイズ|2048|
|イテレーション回数/エポック|4|
|Clip Rangeのε|0.2|
|GAE算出でのλ|0.95|
|GAE算出での報酬の割引率γ|0.99|

※報酬設計の変更　～ gym.Wrapperのサブクラス ～<br>
BipedalWalkerは、失敗時（転倒時）、-100という即時報酬を返してエピソードを終了します。<br>
このように一連の経験データ中に数値規模が著しく大きい即時報酬がポツンポツンとある場合、前述した「即時報酬の標準化」を通すと、他の標準化報酬の数値規模が著しく小さくなってしまいます。<br>
よって、gym.Wrapperのサブクラスを作成し、以下のような即時報酬を返すようにしました。<br>
※このサブクラスを使用するのは訓練時の経験データ収集時のみであり、NN更新後の1エピソード試行に使用するのはオリジナルのgym.Wrapperです。よって、下記グラフのScoreはそのまま見ることができます。
|事象|即時報酬|
|      :---:     |      :---:      | 
|エピソード途中のステップ|オリジナルのgym.Wrapperのrewardと同じ|
|エピソード終端　成功時<br>（2000ステップ転倒しなかった）|+1|
|エピソード終端　失敗時<br>（2000ステップもたず転倒した）|-1|

#### ScoreとLoss

＜Score＞<br>
![BipedalWalker_score](https://user-images.githubusercontent.com/52105933/100704671-35ec3080-33e9-11eb-8362-f8508445bab9.png)

＜Loss＞<br>
![BipedalWalker_loss](https://user-images.githubusercontent.com/52105933/100704730-5320ff00-33e9-11eb-8555-44b2665c268e.png)

#### 考察

こちらは、あまりうまくいきませんでした。<br>
ある時点まではscoreは伸びるが、以降は突然scoreが急降下する、という挙動でした。<br>
これは、実は何度やっても同じでした。<br>
同じタイミングでCriticのlossも跳ね上がっていることから、経験データ内の稼得即時報酬がいきなり下がったことがわかります。<br>
Agentが一体しかない場合は、経験データの”質”が訓練成果に直結しますが、起こっていることはまさにこれだと推測できます。<br>
もしAgentが複数ある場合、どれか一体の経験データの質が悪化しても、残りのAgentの経験データでカバーできます。<br>
というわけで、MountainCarと同様、並列化Agentによる経験データ収集を行っていないことが原因、と思われます。

結局、PPOに限らず、<br>
**経験データ（Trajectory）を収集し、それを使用してNNの最適化を行う、という手法を取る場合は、並列化Agentによる経験データ収集が必要**<br>
ということだと思います。<br>

<br><br>

## 物理構成

PPOAgentTrainer.py　・・・Trainer<BR>
Agent.py　・・・Agent<BR>
common/<br>
&nbsp;└funcs.py　・・・ユーティリティ関数など<br>
&nbsp;└env_wrappers.py　・・・gym.Wrapperの各種サブクラス<br>
<br>

![物理構成_70](https://user-images.githubusercontent.com/52105933/100743622-e2490980-341f-11eb-98ce-c7c10d18d438.png)

<br><br><br>


※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。