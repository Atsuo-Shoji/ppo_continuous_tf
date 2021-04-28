# PPO（Proximal Policy Optimization）（TensorFlow2.3.0を使用）


## 概要

PPO（Proximal Policy Optimization）をTensorflow2.3.0で実装しました。<br>
    
本モデルは、行動が連続値を取る環境を対象としています。<BR>
本稿では、OpenAI GymのBipedalWalkerを使用しています。<br>

※PPOでは普通に行われる並列化Agentによる経験データ（Trajectory）収集は行っていません。Agentは単体です。<BR>

※理論の説明は基本的にしていません。他のリソースを参考にしてください。<br>
&nbsp;&nbsp;ネットや書籍でなかなか明示されておらず、私自身が実装に際し情報収集や理解に不便を感じたものを中心に記載しています。

<br>
    
### 使用した環境
本稿の実験においては、以下の環境を使用しています。<br>

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|2足歩行して遠くまで行く|

#### 未訓練モデル/訓練済モデルでplayした結果の比較

| 未訓練モデルでPlay<br>すぐ転倒し前に進めない | | PPO訓練済モデルでPlay<br>2足歩行してある程度のところまで前進 |
|      :---:       |     :---:      |     :---:      |
|![BipedalWalker_beginner_66s](https://user-images.githubusercontent.com/52105933/95009942-9c123c80-0660-11eb-9cb1-b5ee0a2a90f7.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![bipwalker_ver20_wrap01_pos1_neg1_10it_entcof01_202103030756_st717_r191](https://user-images.githubusercontent.com/52105933/113482775-2128c200-94db-11eb-8e44-d4b711a915b8.gif)|

<br><br>

## 実装の要点

### 全体の構成

Trainerは、AgentをPPOメソッドに従い訓練します。<br>
Agentの中身は、Actor-Criticです（両者独立したNN）。<BR>
PPOでは普通に行われる並列化Agentによる経験データ収集は行っていません。Agentは単体です。

![全体構成図_80](https://user-images.githubusercontent.com/52105933/110124918-0db80780-7e06-11eb-9b60-61979ea59870.png)

#### Agentの構成

Actor-Criticで、ActorとCriticは別々のNNにしています。<BR>

Actor側では、行動はガウス方策に従います。<br>
よって、Actorの出力は、正規分布の平均μと標準偏差σとなります。<br>
行動の次元数をK(=4)とすると、平均μと標準偏差σの1セットがK(=4)個ある、ということになります。<br>

Critic側は、単純に状態価値関数値を出力するだけです。<br>

Actor側、Critic側双方で、GAE（Generalized Advantage Estimation）を使用しています。

![NN構成図_70](https://user-images.githubusercontent.com/52105933/110185584-13d5d480-7e56-11eb-8f29-71c899ff557d.png)

<br>

### 訓練処理

#### 流れ

1エポック内で、以下のことを上から順に行います。<br>
- （初回エポックのみ）経験データ（Trajectory）収集<br>
- Agentの訓練<br>
- 経験データ（Trajectory）収集<br>
このエポックの評価と、次のエポックでのAgentの訓練のため

これを複数エポック繰り返します。

![訓練処理の時間軸_70](https://user-images.githubusercontent.com/52105933/110188042-be9dc100-7e5d-11eb-9fe9-613a1378ea9b.png)

#### 報酬設計の変更　～ gym.Wrapperのサブクラス

BipedalWalkerは、失敗時（転倒時）、-100という即時報酬を返してエピソードを終了します。<br>
このように一連の経験データ中に数値規模が著しく大きい即時報酬がポツンポツンとある場合、それは実質”外れ値”となります。<br>
後述する「即時報酬の標準化」を通すと、他の標準化報酬の数値規模が著しく小さくなってしまいます。<br>
よって、gym.Wrapperのサブクラスを作成し、以下のような即時報酬を返すようにしました。
|事象|即時報酬|
|      :---:     |      :---:      | 
|エピソード途中のステップ|オリジナルのgym.Wrapperのrewardと同じ|
|エピソード終端　成功時<br>（2000ステップ転倒しなかった）|+1|
|エピソード終端　失敗時<br>（2000ステップもたず転倒した）|-1|

#### ratio（「rt(Θ)」）

ActorのLossをclipするのに使用されるratioは、以下のように算出しています。

![ratio説明_80](https://user-images.githubusercontent.com/52105933/110198650-6d122800-7e97-11eb-83e2-ae9f3441086f.png)

#### 即時報酬の標準化

稼得した即時報酬の標準化を行っています。<BR>
この標準化された即時報酬を使用して、GAE（Generalized Advantage Estimation）の算出をします。<BR>
ただし、この標準化において、平均を引き算しません。<BR>
平均を引くと、実際の即時報酬とはプラスマイナスの符号が逆になってしまう標準化報酬が出てきてしまうからです。<BR>

<br><br>

## 訓練結果

### 方策エントロピー項無し　の場合

まずは、方策エントロピー項無しで、訓練してみました。

#### 訓練の設定値など

|項目|値など|
|      :---:     |      :---:      | 
|エポック数|500|
|経験データサイズ|20480|
|バッチサイズ|2048|
|イテレーション回数/エポック|10|
|Clip Rangeのε|0.2|
|GAE算出でのλ|0.95|
|GAE算出での報酬の割引率γ|0.99|
|方策エントロピー項の係数c|**0**|

#### 各エポックでの訓練成果の評価グラフ

![epoch推移グラフ群_80](https://user-images.githubusercontent.com/52105933/110078652-674f1080-7dcb-11eb-8512-a68a0083aba4.png)

#### 考察

以下のような結果になりました。<br>
- 1エポックあたりの稼得報酬は、途中まで増加した後、不安定になっている<br>
- 稼得報酬の1エピソード平均は、ゆるやかに増加している<br>
- ステップ数の1エピソード平均は、あるタイミングで少し増加した後、不安定になっている

また、グラフからは分かりにくいですが、ある程度訓練が進むと、「ステップ数エピソード平均が大きいと、そのエポックの稼得報酬合計が少な目」という傾向になっています（グラフを跨いだ縦の点線を参考に）。<br>
これらのことから、1ステップあたりの報酬はマイナスになることが多く、「傷口が広がる前に早めにコケて稼得報酬合計の高さを維持する」ことを学んでしまったのか、と推測しました。<br>
この状況を打破するには、**「1つの方策に凝り固まることなく、いろんな手を打たせる」のが有効か、と推測**しました。<br>
こうして、下記の方策エントロピー項を追加することにしました。<br>
<br>


### 方策エントロピー項有り　の場合

次に、方策エントロピー項有りで、訓練してみました。

#### 方策エントロピー項

Actor（Policy側）のLossにおいて、以下のように、現在の方策のエントロピーをマイナスします。<BR>
マイナスするのは、方策が確定的でない場合に比べて、方策が確定的である場合のLossが大きくなるようにするためです。<br>

![方策エントロピー補正_80](https://user-images.githubusercontent.com/52105933/110184751-fef84180-7e53-11eb-9d89-d9d88fdd010e.png)

#### 訓練の設定値など

方策エントロピー項の係数c以外、全て同じです。

|項目|値など|
|      :---:     |      :---:      | 
|エポック数|500|
|経験データサイズ|20480|
|バッチサイズ|2048|
|イテレーション回数/エポック|10|
|Clip Rangeのε|0.2|
|GAE算出でのλ|0.95|
|GAE算出での報酬の割引率γ|0.99|
|方策エントロピー項の係数c|**0.1**|

#### 訓練成果の動画

（500エポック終了時点のもの）<BR>
![bipwalker_ver20_wrap01_pos1_neg1_10it_entcof01_202103030756_st717_r191](https://user-images.githubusercontent.com/52105933/113482775-2128c200-94db-11eb-8e44-d4b711a915b8.gif)

#### 各エポックでの訓練成果の評価グラフ

![epoch推移グラフ群_80_W](https://user-images.githubusercontent.com/52105933/110111228-574b2700-7df3-11eb-8497-d517170b278f.png)

#### 考察

以下のような結果になりました。<br>
- 1エポックあたりの稼得報酬<br>
方策エントロピー項有りの方は、常に無しより多く、且つ増加傾向が止まることが無い。100エポック付近から増加スピードが遅くなる。<br>
- 稼得報酬の1エピソード平均<br>
方策エントロピー項有りの方は、ほぼ常に無しより多く、増加スピードも速い。<br>
- ステップ数の1エピソード平均<br>
方策エントロピー項有りの方は、無しに比べて高かったり低かったりしているが、最終的には高くなっている。<BR>
さらに、大局的に見て、100エポック付近から増加傾向を常に維持している。<BR>
- Actor（Policy側）のLoss<br>
方策エントロピー項の有無では変わりがない。<br>
- Critic（Value側）のLoss<br>
方策エントロピー項有りの方は、無しに比べて、100エポック付近からは常に高く、且つ大局的に見て増加傾向がある。

「1エポックあたりの稼得報酬」「稼得報酬の1エピソード平均」「ステップ数の1エピソード平均」の3つすべてがほぼ最高値という3冠王的エポック（グラフの緑の〇）まで出てきています。<BR>
**「報酬とステップ数をともに大きく」というBipedalWalkerのタスクの目的に訓練の方向が正しく向いており、このまま訓練を継続していたならば、さらに報酬とステップ数を伸ばすことができた**と思います。<br>
「ステップ数エピソード平均が大きいと、そのエポックの稼得報酬合計が少な目」という傾向は、概ね解消されているように見えます（ただそのようになっているエポックもまだ残っています（グラフの赤の〇））。<br>

Critic（Value側）のLossがむしろ増加傾向になった、というのは、稼得報酬が増えたからでは、と思います。<br>
Criticでは、Trajectory収集時のCritic出力値 + それをもとに算出したGAE　が教師信号です。<br>
よって、Criticのパラメーター更新時のCriticのLossはGAEの2乗に近い値になっているだろう、と推測できます。つまり、GAEが大きくなるほどCriticのLossも大きくなる傾向になるはずです。<BR>

訓練試行回数が少なく（方策エントロピー項有りと無しとでそれぞれ2回）、断定はできませんが、**「方策エントロピー項の効果が見受けられる」とは言えそう**です。<br>
ちなみに、その各々の2回は、いずれも方策エントロピー項有りが無しよりも結果は良かったです。

<br><br>
## 実行確認環境と実行の方法

本リポジトリは、他者に提供するためまたは実行してもらうために作ったものではないため、記載しません。<br>
（tensorflow-probabilityと、それに適合するバージョンのTensorFlow2.xが最低限必要です。）<br>
<br>

## ディレクトリとモデルの構成

training.ipynb　・・・実際にモデルの訓練をしているノートブック<BR>
PPOAgentTrainer.py　・・・Trainer<BR>
Agent.py　・・・Agent<BR>
common/<br>
&nbsp;└funcs.py　・・・ユーティリティ関数など<br>
&nbsp;└env_wrappers.py　・・・gym.Wrapperの各種サブクラス<br>
<br>

![物理構成_70](https://user-images.githubusercontent.com/52105933/100743622-e2490980-341f-11eb-98ce-c7c10d18d438.png)

<br><br>


※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。