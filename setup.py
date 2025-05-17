from setuptools import setup

setup(
    name="ai-lab-assessment",
    version="1.0.0",
    install_requires=[
        "flask==2.0.3",
        "Werkzeug==2.0.3",
        "numpy==1.24.3",
        "pandas==1.5.3",
        "pillow==9.5.0",
        "tflite-runtime==2.5.0",
        "openpyxl==3.0.10",
        "gunicorn==20.1.0"
    ],
    python_requires=">=3.9",
) 