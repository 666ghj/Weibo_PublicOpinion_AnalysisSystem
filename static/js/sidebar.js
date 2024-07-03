(function (jQuery) {
    "use strict";
    const urlParams = new URLSearchParams(window.location.search);
    const sidebar = urlParams.get('sidebar');
    if (sidebar !== null) {
        $('.iq-sidebar').removeClass('sidebar-dark', 'sidebar-light')
        switch (sidebar) {
            case "0":
                $('.iq-sidebar').addClass('sidebar-dark')
            break;
            case "1":
                $('.iq-sidebar').addClass('sidebar-light')
            break;
            default:
                $('.iq-sidebar').removeClass('sidebar-dark').removeClass('sidebar-light')
                break;
        }
    }
})(jQuery)