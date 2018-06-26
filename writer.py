def writer(path, outputlist, writerType):
    with open("%s" % path , writerType) as f:#格式化字符串还能这么用！
        for i in outputlist:
            f.write(i)
