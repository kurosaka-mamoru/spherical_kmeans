package = "spkmeans"
version = "scm-1"

source = {
   url = "git@github.com:kurosaka-mamoru/spherical_kmeans.git",
   tag = "master"
}

description = {
   summary = "spherical k-means",
   detailed = [[
   	    spherical k-means
   ]],
   homepage = "git@github.com:kurosaka-mamoru/spherical_kmeans.git"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}