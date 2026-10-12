% Writes a value as pretty-printed JSON: cmp_json(file,value).
function cmp_json(file,value)
fid=fopen(file,'w');
if fid<0, error(['cannot write ' file]); end
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
fclose(fid);
end
