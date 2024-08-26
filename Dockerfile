FROM aiops/mls:backbone

COPY ./ /root/ai-module/mlops-mls

WORKDIR /root/ai-module/mlops-mls
RUN chmod -R 777 ./*