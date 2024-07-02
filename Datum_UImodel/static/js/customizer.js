(function (jQuery) {
    "use strict";
    // data-mode="click" for using event
    // data-dark="false" for property
    // icon class // la-sun // la-moon
    const storageDark = localStorage.getItem('dark')
    if($('body').hasClass('dark')){
        changeMode('true');
    } else {
        changeMode('false');
    }
    if (storageDark !== 'null') {
        changeMode(storageDark)
    }
    jQuery(document).on("change", '.change-mode input[type="checkbox"]' ,function (e) {
        const dark = $(this).attr('data-active');
        if (dark === 'true') {
            $(this).attr('data-active','false')
        } else {
            $(this).attr('data-active','true')
        }
        changeMode(dark)
    })
    function changeMode (dark) {
        const body = jQuery('body')
        if (dark === 'true') {
            // $('[data-mode="toggle"]').find('i.a-right').removeClass('ri-sun-line');
            // $('[data-mode="toggle"]').find('i.a-left').addClass('ri-moon-clear-line');
            $('#dark-mode').prop('checked', true).attr('data-active', 'false')
            $('.darkmode-logo').removeClass('d-none')
            $('.light-logo').addClass('d-none')
            body.addClass('dark')
            dark = true
        } else {
            // $('[data-mode="toggle"]').find('i.a-left').removeClass('ri-moon-clear-line');
            // $('[data-mode="toggle"]').find('i.a-right').addClass('ri-sun-line');
            $('#dark-mode').prop('checked', false).attr('data-active', 'true');
            $('.light-logo').removeClass('d-none')
            $('.darkmode-logo').addClass('d-none')
            body.removeClass('dark')
            dark = false
        }
        updateLocalStorage(dark)
        const event = new CustomEvent("ChangeColorMode", {detail: {dark: dark} });
        document.dispatchEvent(event);
    }
    function updateLocalStorage(dark) {
        localStorage.setItem('dark', dark)
    }
    
})(jQuery)