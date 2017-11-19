TAG ?= local-dev
DOCKER_REPOSITORY := sergii/text-detection

.PHONY: build
build:
	docker build -t ${DOCKER_REPOSITORY}:${TAG} .


.PHONY: run
run:
	docker run -it \
		-p 8769:8769 \
		-v ${PWD}/../east_icdar2015_resnet_v1_50_rbox/:/app/east_icdar2015_resnet_v1_50_rbox/ \
		${DOCKER_REPOSITORY}:${TAG}