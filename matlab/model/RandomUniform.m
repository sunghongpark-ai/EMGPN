function U = RandomUniform(Stream, NumRow, NumCol)

if isa(Stream, 'RandStream')
    U = rand(Stream, NumRow, NumCol);
else
    U = rand(NumRow, NumCol);
end
end
