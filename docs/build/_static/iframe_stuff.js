// This piece of code removes the unnecessary scrolling option for pyvista's iframes
document.addEventListener('DOMContentLoaded', function() {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                var iframes = document.getElementsByClassName('pyvista');
                for (var i = 0; i < iframes.length; i++) {
                    iframes[i].setAttribute('scrolling', 'no');
                }
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});