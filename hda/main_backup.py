def main():
    config = load_config("config.yaml")
    batch_size, versions, time_steps, diodes = get_dataset_shape(config)

    preprocessor = Preprocessor("config.yaml")
    train_dataset, val_dataset, test_dataset = preprocessor.run()

    y_train = np.concatenate(
        [np.array([label.numpy()]) for _, label in train_dataset.unbatch()],
        axis=0)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    # Instantiate the model
    model = TimeSeriesModelGRU(versions, time_steps, diodes, 7,
                               config['convolutions_conf'],
                               config['dense_conf'], num_gru_units=128)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Provide the input shape to build the model (necessary for model.summary())
    model.build(input_shape=(batch_size, versions, time_steps, diodes))

    # Model summary to check the architecture
    model.summary()

    # Wrap the model with WandbKerasModel
    wandb_model = WandbKerasModel(
        model=model, project_name="hda_small", config={}, entity="bdma"
    )

    # Compile and fit the model
    wandb_model.compile_and_fit(
        compile_args={
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        fit_args={
            "x": x_train,
            "y": y_train,
            "epochs": 3,
            "batch_size": 32,
            "validation_data": (x_test, y_test),
            "experiment_name": "mnist_simple_nn_3epochs",
        },
    )

    # Training the model with validation
    # Training the model with validation
    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=val_dataset,
        # class_weight=class_weight_dict
    )

    # Evaluate the model on the test set and print the confusion matrix
    print("Evaluating on test set...")
    y_true = []
    y_pred = []

    # labels_val = []
    # for _, labels in val_dataset:
    #     labels_val.extend(labels.numpy().flatten())
    # print(Counter(labels_val))

    # Assuming your test_dataset yields (features, labels)
    for features, labels in test_dataset:
        preds = model.predict(features)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(np.argmax(preds, axis=-1).flatten())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Create an intermediate model using the functional API
    inputs = tf.keras.Input(shape=(versions, time_steps, diodes))
    x = model.multi_version_conv(
        inputs)  # Use the WindowsConvolutionLayer as a functional layer
    intermediate_model = tf.keras.Model(inputs=inputs, outputs=x)

    # Prepare a sample input
    sample_input = tf.random.normal([1, versions, time_steps, diodes])

    # Pass the sample input through the model to ensure it's built
    _ = model(sample_input)

    # Get the feature maps for the sample input
    feature_maps = intermediate_model.predict(sample_input)
    print(feature_maps.shape)

    # Assuming the shape of feature_maps is (1, versions, time_steps, diodes, filters)
    # And you want to plot for the first diode
    diode_index = 0  # Index for the first diode

    # Plot the feature maps for the selected diode across all versions
    print(f"Feature Maps for Diode {diode_index + 1}")
    plot_feature_maps(feature_maps, diode_index=diode_index)


if __name__ == '__main__':
    main()
