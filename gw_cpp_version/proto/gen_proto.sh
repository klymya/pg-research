#!/bin/bash

protoc --proto_path=. --cpp_out=. ./sensor_payload.proto