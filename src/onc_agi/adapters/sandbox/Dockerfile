# Analysis sandbox for the ONC-AGI standard harness: Python with common
# statistics libraries. The compose file disables networking at run time.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.* pandas==2.* scipy==1.* scikit-learn==1.* statsmodels==0.14.*
RUN mkdir -p /data
WORKDIR /data
