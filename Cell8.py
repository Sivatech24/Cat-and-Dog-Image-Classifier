# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.n // batch_size,
    epochs=100,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.n // batch_size
)

# Output
print("Model trained successfully.")
