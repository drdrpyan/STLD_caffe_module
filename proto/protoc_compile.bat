set IMPORT_DIR=D:/library/caffe_env/caffe/include
set SRC_DIR=D:/workspace/TLR/caffe/proto
set DST_DIR=D:/workspace/TLR/caffe/proto
set PROTO_FILE=caffe_extend.proto

D:/dev_tools/protobuf/protoc-3.1.0-win32/bin/protoc -I=%IMPORT_DIR% -I=%SRC_DIR% --cpp_out=%DST_DIR% %SRC_DIR%/%PROTO_FILE%