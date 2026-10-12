% Reads an option with a default: value=cmp_option(options,'npoints',512).
function value=cmp_option(options,name,default)
if isstruct(options)&&isfield(options,name)
    value=options.(name);
else
    value=default;
end
end
