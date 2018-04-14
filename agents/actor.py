from keras import layers, models, optimizers
from keras import backend as K


class Actor():

    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.dropout_rate = 0.2
        self.hidden_size = 16
        self._build_model()

    def _build_model(self):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=self.hidden_size, activation='relu')(states)
        net = layers.Dropout(self.dropout_rate)(net)
        net = layers.Dense(units=self.hidden_size * 2, activation='relu')(net)
        net = layers.Dropout(self.dropout_rate)(net)
        net = layers.Dense(units=self.hidden_size, activation='relu')(net)
        net = layers.Dropout(self.dropout_rate)(net)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(
            units=self.action_size,
            activation='sigmoid',
            name='raw_actions'
        )(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(
            lambda x: (x * self.action_range) + self.action_low,
            name='actions'
        )(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights,
            loss=loss,
        )
        self.train = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op
        )
