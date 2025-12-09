import os
import tensorflow as tf
from clr_bnn import CLR_BNN
from train_clr import CLR_BNN_Trainer
from dataloader import create_tf_dataset, NuScenesGenerator

# Configuración
# Ajusta esto a donde tengas tu 'v1.0-mini' o 'v1.0-trainval'
NUSC_ROOT = "/mnt/c/Users/USUARIO/Desktop/datasets/nuscenes/v1.0-mini"
INPUT_SHAPE = (320, 320)
BATCH_SIZE = 2
EPOCHS = 10

if __name__ == "__main__":
    print(f"🚀 Iniciando Entrenamiento CLR-BNN en NuScenes...")
    print(f"   - Root: {NUSC_ROOT}")
    print(f"   - Shape: {INPUT_SHAPE}")
    print(f"   - Batch Size: {BATCH_SIZE}")

    # 1. Contar muestras reales para ajustar el KL Weight
    # Instanciamos el generador solo para pedirle la longitud (__len__)
    print("📊 Calculando tamaño del dataset...")
    temp_gen = NuScenesGenerator(NUSC_ROOT, input_shape=INPUT_SHAPE)
    TOTAL_SAMPLES = len(temp_gen)
    print(f"✅ Total Muestras: {TOTAL_SAMPLES}")

    # 2. Crear el DataLoader de TensorFlow
    print("🔄 Creando Pipeline de Datos...")
    train_ds = create_tf_dataset(NUSC_ROOT, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE)

    # 3. Inicializar Modelo
    print("🧠 Construyendo Modelo...")
    model = CLR_BNN(num_classes=10, input_shape=INPUT_SHAPE)
    
    # 4. Inicializar Entrenador
    # Usamos el TOTAL_SAMPLES real para que el KL Loss se escale perfecto
    trainer = CLR_BNN_Trainer(
        model, 
        initial_lr=1e-4, # LR suave para empezar
        train_dataset_size=TOTAL_SAMPLES 
    )

    # 5. Entrenar
    print("🔥 ¡Empezando a Entrenar!")
    try:
        trainer.fit(train_ds, epochs=EPOCHS)
        
        # Guardar pesos al final
        os.makedirs("weights", exist_ok=True)
        model.save_weights("weights/clr_bnn_final.h5")
        print("\n💾 Pesos guardados en 'weights/clr_bnn_final.h5'")
        
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido manualmente.")
        model.save_weights("weights/clr_bnn_interrupted.h5")
        print("💾 Pesos parciales guardados.")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()