
    $(document).ready(function () {
        $('#basic').on('click', function () {
            Swal.fire('Hello! This is a Basic Message.')
    	});
        $('#basic-title').on('click', function () {
            Swal.fire(
                'The Internet?',
                'That thing is still around?',
                'question'
            )
        });
        $('#success').on('click', function () {
            Swal.fire({                
                icon: 'success',
                title: 'Your work has been saved',
                showConfirmButton: false,                
            })
        });
        $('#info').on('click', function () {
            Swal.fire({                
                icon: 'info',
                title: 'Good Job!',
                showConfirmButton: false,                
            })
        });
        $('#warning').on('click', function () {
            Swal.fire({                
                icon: 'warning',
                title: 'Changes are not saved',
                showConfirmButton: false,
                
            })
        });
        $('#danger').on('click', function () {
            Swal.fire({
                icon: 'error',
                title: 'OOps!!',
                text: 'Something went Wrong',
                showConfirmButton: false,                
            })
        });
        $('#confirmation').on('click', function () {
            Swal.fire({
                title: "Are you sure?",
                text: "Once deleted, you will not be able to recover this imaginary file!",
                icon: "warning",
                buttons: true,
                dangerMode: true,
            })
            .then((willDelete) => {
                if (willDelete) {
                    Swal.fire("Poof! Your imaginary file has been deleted!", {
                        icon: "success",
                    });
                } else {
                    Swal.fire("Your imaginary file is safe!");
                }
            });
        });
        $('#custom-buttons').on('click', function () {
            Swal.fire("A wild Pikachu appeared! What do you want to do?", {
                buttons: {
                    cancel: "Run away!",
                    catch: {
                        text: "Throw Pokéball!",
                        value: "catch",
                    },
                    defeat: true,
                },
            })
            .then((value) => {
                switch (value) {

                    case "defeat":
                        Swal.fire("Pikachu fainted! You gained 500 XP!");
                        break;

                    case "catch":
                        Swal.fire("Gotcha!", "Pikachu was caught!", "success");
                        break;

                    default:
                        Swal.fire("Got away safely!");
                }
            });
        });
        $('#ajax-request').on('click', function () {
            Swal.fire({
                text: 'Search for a movie. e.g. "La La Land".',
                content: "input",
                button: {
                    text: "Search!",
                    closeModal: false,
                },
            })
            .then(name => {
                if (!name) throw null;

                return fetch(`https://itunes.apple.com/search?term=${name}&entity=movie`);
            })
            .then(results => {
                return results.json();
            })
            .then(json => {
                const movie = json.results[0];

                if (!movie) {
                    return Swal.fire("No movie was found!");
                }

                const name = movie.trackName;
                const imageURL = movie.artworkUrl100;

                Swal.fire({
                    title: "Top result:",
                    text: name,
                    icon: imageURL,
                });
            })
            .catch(err => {
                if (err) {
                    Swal.fire("Oh noes!", "The AJAX request failed!", "error");
                } else {
                    Swal.fire.stopLoading();
                    Swal.fire.close();
                }
            });
        }); 
        $('#form-input').on('click', function () {
            Swal.fire("Write something here:", {
                content: "input",
            })
            .then((value) => {
                Swal.fire(`You typed: ${value}`);
            });
        });      
    });