

class MAMLTransformer():
    def maml_loss(self,model,X_train,Y_train,X_val,Y_val,inner_lr = 0.01):
        with tf.GradientTape() as tape: 
            predictions = model(X_train)
            loss = tf.keras.losses.binary_crossentropy(Y_train,predictions)
        grads = tape.gradient(loss,model.trainable_variables)
        k = [w-inner_lr*g for w,g in zip(model.traianble_variables,grads)]
        model.set_weights(k)

        val_predictions = model(X_val)
        val_loss = tf.keras.losses.binary_crossentropy(Y_val,val_predictions)
        return val_loss
    
    def train_transformer_model(self,X,Y):
        time_steps = 10
        X_train,X_test,Y_train,Y_test = self.create_train_test_split_group(X,Y,split_ratio = 0.8)
        X_train = self.prepare_transfomer_data(X_train,time_steps)
        X_test = self.prepare_transformer_data(X_test,time_steps)
        Y_train = Y[time_steps:len(X_train) + time_steps]
        Y_test = Y[len(X_train)+ time_steps:]

        input_shape = X_train.shape[1:]
        model = self.build_transformer_model(input_shape, head_size = 256,num_heads = 4, ff_dim = 4, num_transfomer_blocks = 4, mlp_units = [128],mlp_dropout = 0.4,dropout = 0.25)
        optimizer = tf.keras.optimizers.Adam()

        for _ in range(100):
            inner_loss = self.maml_loss(model,X_train,Y_train,X_test,Y_test)
            optimizer.minimize(inner_loss,model.trainable_variables)

        return model