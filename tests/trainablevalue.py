import tensorflow as tf

class ParamModel(tf.keras.Model):
    """Actor-Criticモデル (連続行動空間対応)"""
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim # action_dimをインスタンス変数として保持

    def build(self, input_shape):
        # buildメソッド内でadd_weightを使用
        self.log_std = self.add_weight(
            name='log_std',
            shape=(self.action_dim,),
            initializer='zeros',
            trainable=True
        )
        # 親クラスのbuildを呼び出す
        super().build(input_shape)

    def call(self, inputs):
        # モデルがビルドされた後に、self.log_stdが利用可能になる
        return inputs + self.log_std

param = ParamModel(1)

# callメソッドを一度実行すると、buildが自動的に呼び出され、変数が作成される
_ = param(tf.zeros((1, 1)))

print('trainable_variables=', param.trainable_variables)
print(tf.__version__)