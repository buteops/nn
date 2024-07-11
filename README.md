### fromtenets: from neural nets to mlops

> [!IMPORTANT]
> the way to learn are from books to implementations. Target 50 books

All code are written with plain python (.py), you must know about Modular Programming. All Production-Ready are used as feature with API behaviour (more or less). You think notebook are served to production-use?

**Need an Advice?**

The Advice to Learn from [Andrej Karpathy][0] or from [George Hotz][1]
> Both of two are kinda [contradictive][2], but you can take it as auxiliary

**10000 hours Syllabus**
> The books reference will be added later :)

| **Level**                | **Sub-Area**                              | **Description**                                                    | **Topics/Tools/Technologies**                                                                                              | **Estimated Hours** |
|--------------------------|-------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------|
| **Beginner**             | **Supervised Learning**                   | Learning from labeled data to make predictions.                    | Linear Regression, Decision Trees, SVM, Random Forests, Scikit-learn, XGBoost, LightGBM                                    | 800                 |
|                          | **Unsupervised Learning**                 | Learning from unlabeled data to find patterns.                     | K-Means, PCA, Hierarchical Clustering, Autoencoders, Scikit-learn, TensorFlow, PyTorch                                     | 600                 |
|                          | **Data Engineering**                      | Data ingestion, transformation, and storage.                       | Apache Kafka, Apache Nifi, Flume, Apache Spark, Apache Beam, Talend, Hadoop, Amazon S3, Google BigQuery                   | 500                 |
|                          | **Model Development**                     | Model creation, training, and validation.                          | Jupyter, VS Code, TensorFlow, PyTorch, Scikit-learn, Git, DVC                                                             | 800                 |
|                          | **Neural Networks**                       | Study of artificial neurons and architectures.                     | CNNs, RNNs, LSTMs, GANs, Transformers, TensorFlow, PyTorch, Keras                                                           | 1000                |
|                          | **Natural Language Processing (NLP)**     | Processing and understanding human language.                       | Tokenization, Named Entity Recognition, Sentiment Analysis, NLTK, SpaCy, Hugging Face Transformers                         | 800                 |
|                          | **Computer Vision**                       | Enabling machines to interpret and understand visual data.         | Image Classification, Object Detection, Image Segmentation, OpenCV, TensorFlow, PyTorch                                    | 800                 |
|                          | **CI/CD**                                 | Automating build, test, and deployment phases.                     | Jenkins, GitLab CI/CD, CircleCI, Apache Airflow, Kubeflow Pipelines                                                       | 500                 |
|                          | **Model Management**                      | Managing versions, metadata, and lifecycle of models.              | MLflow, DVC, Neptune.ai, TFX, Metaflow                                                                                     | 400                 |
|                          | **Infrastructure Management**             | Handling the underlying infrastructure for MLOps.                  | Terraform, Ansible, AWS, GCP, Azure, Kubernetes, OpenShift                                                                  | 500                 |
|                          | **Advanced Courses**                      | Specialized and in-depth courses on ML & AI.                       | Coursera, edX, Udacity, Fast.ai, Andrew Ng's Deep Learning Specialization, CS231n (Stanford), CS224n (Stanford)            | 400                 |
| **Intermediate**         | **Reinforcement Learning**                | Learning through rewards and penalties.                            | Q-Learning, Deep Q-Networks, Policy Gradients, OpenAI Gym, Ray RLlib, TensorFlow Agents                                     | 600                 |
|                          | **Model Deployment**                      | Deploying models to production environments.                       | Docker, Kubernetes, Docker Swarm, TensorFlow Serving, TorchServe, FastAPI                                                  | 600                 |
|                          | **Monitoring**                            | Tracking model performance and system health.                      | ELK Stack (Elasticsearch, Logstash, Kibana), Prometheus, Grafana, Nagios, Seldon Core                                      | 500                 |
|                          | **Embedded Machine Learning**             | Implementing ML algorithms on resource-constrained devices.        | TensorFlow Lite, PyTorch Mobile, CoreML, Edge Impulse, TFLite Micro, ARM CMSIS-NN                                          | 800                 |
|                          | **Edge AI**                               | Deploying AI models on edge devices for real-time processing.      | Nvidia Jetson, Intel Movidius, Google Coral, TensorFlow Lite, OpenVINO, AWS Greengrass                                      | 800                 |
|                          | **Low-Power Machine Learning**            | Developing ML models efficient in terms of power consumption.      | TinyML, Quantized Neural Networks (QNNs), ARM Cortex-M, RISC-V, Microcontrollers                                            | 600                 |
|                          | **Real-Time Processing**                  | Ensuring ML models can process data in real-time on embedded systems.| Real-time Operating Systems (RTOS), Stream Processing, FreeRTOS, Zephyr OS, Apache NiFi                                    | 500                 |
| **Advanced**             | **Model Optimization**                    | Reducing model size and improving efficiency.                      | Quantization, Pruning, Knowledge Distillation, TensorFlow Model Optimization Toolkit, ONNX Runtime                         | 500                 |
|                          | **Sensor Fusion**                         | Combining data from multiple sensors for more accurate predictions. | Kalman Filters, Bayesian Networks, Arduino, Raspberry Pi, Nvidia Jetson                                                     | 400                 |
|                          | **Connectivity**                          | Robust communication between embedded devices and cloud.           | MQTT, CoAP, LoRaWAN, BLE, AWS IoT, Azure IoT, Google Cloud IoT Core                                                         | 400                 |
|                          | **Security**                              | Ensuring data privacy and security in embedded AI applications.    | Secure Boot, Encryption, Anomaly Detection, Arm TrustZone, Secure Elements                                                  | 400                 |
|                          | **Ethics and Fairness**                   | Ensuring ethical AI and fairness in ML models.                     | Bias Mitigation, Explainability, Fairness Metrics, IBM AI Fairness 360, Google What-If Tool                                 | 300                 |
|                          | **Advanced ML Topics**                    | More complex and specialized areas in ML.                          | Meta-Learning, Federated Learning, Few-Shot Learning, Transfer Learning, TFLite, Edge TPU, ONNX                             | 600                 |
|                          | **Explainable AI (XAI)**                  | Making AI decisions interpretable by humans.                       | SHAP, LIME, Explainable Boosting Machine (EBM), InterpretML                                                                  | 400                 |
|                          | **AutoML**                                | Automated model selection and hyperparameter tuning.               | AutoKeras, TPOT, H2O.ai, Google Cloud AutoML                                                                                | 400                 |
|                          | **Adversarial Machine Learning**          | Techniques to make models robust against adversarial attacks.      | FGSM, PGD, Adversarial Training, CleverHans                                                                                 | 400                 |
|                          | **Scalable Machine Learning**             | Approaches for scaling ML algorithms and infrastructure.           | Apache Spark MLlib, Dask-ML, TensorFlow on Kubernetes, Horovod                                                              | 500                 |
|                          | **Graph Neural Networks (GNNs)**          | Learning from data structured as graphs.                           | Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), DGL, PyTorch Geometric                                | 500                 |
|                          | **Advanced Optimization Algorithms**      | Advanced optimization techniques for ML models.                    | Evolutionary Algorithms, Bayesian Optimization, Hyperopt, Optuna                                                            | 400                 |
|                          | **Data Augmentation and Synthetic Data**  | Increasing the diversity of training data.                         | SMOTE, Data Augmentation, GANs for Synthetic Data, Augmentor, Imgaug                                                        | 400                 |
|                          | **Advanced Courses**                      | Specialized courses focusing on ML & AI for embedded systems.      | Coursera, edX, Udacity, Edge Impulse Learning, TinyML by HarvardX, Embedded Systems by University of Colorado Boulder       | 400                 |


[0]: https://cs.stanford.edu/people/karpathy/advice.html
[1]: https://www.youtube.com/watch?v=NjYICpXJ03M
[2]: https://www.youtube.com/watch?v=lXusHWturrk