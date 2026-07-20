#!/bin/bash

../../../../../../dependencies/nanopb/generator/protoc --nanopb_out=. ./pg.proto

if [ $? != 0 ]; then
    echo "######################################################################"
    echo -e "Make sure that protobuf grpcio-tools modules are installed.\n\tpip install protobuf grpcio-tools"
    echo "######################################################################"
fi
