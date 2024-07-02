/*
Template: Datum - Responsive Bootstrap 4 Admin Dashboard Template
Author: iqonic.design
Design and Developed by: iqonic.design
NOTE: This file contains the styling for responsive Template.
*/

/*----------------------------------------------
Index Of Script
------------------------------------------------

:: Tooltip
:: Fixed Nav
:: Magnific Popup
:: Ripple Effect
:: Sidebar Widget
:: FullScreen
:: Page Loader
:: Counter
:: Progress Bar
:: Page Menu
:: Close  navbar Toggle
:: Mailbox
:: chatuser
:: chatuser main
:: Chat start
:: todo Page
:: user toggle
:: Data tables
:: Form Validation
:: Active Class for Pricing Table
:: Flatpicker
:: Scrollbar
:: checkout
:: Datatables
:: image-upload
:: video
:: dark mode
:: Button
:: Pricing tab
:: SVG Animation
:: Date Picker
:: Choies.js
------------------------------------------------
Index Of Script
----------------------------------------------*/

(function(jQuery) {



    "use strict";

    jQuery(document).ready(function() {

        /*---------------------------------------------------------------------
        Tooltip
        -----------------------------------------------------------------------*/
        jQuery('[data-toggle="popover"]').popover();
        jQuery('[data-toggle="tooltip"]').tooltip();

        /*---------------------------------------------------------------------
        Fixed Nav
        -----------------------------------------------------------------------*/

        $(window).on('scroll', function () {
            if ($(window).scrollTop() > 0) {
                $('.iq-top-navbar').addClass('fixed');
            } else {
                $('.iq-top-navbar').removeClass('fixed');
            }
        });

        $(window).on('scroll', function () {
            if ($(window).scrollTop() > 0) {
                $('.white-bg-menu').addClass('sticky-menu');
            } else {
                $('.white-bg-menu').removeClass('sticky-menu');
            }
        });


       /*---------------------------------------------------------------------
        Sidebar Widget
        -----------------------------------------------------------------------*/

        jQuery(document).on("click", '.side-menu > li > a', function() {
            jQuery('.side-menu > li > a').parent().removeClass('active');
            jQuery(this).parent().addClass('active');
        });

        // Active menu
        var parents = jQuery('li.active').parents('.submenu.collapse');

        parents.addClass('show');


        parents.parents('li').addClass('active');
        jQuery('li.active > a[aria-expanded="false"]').attr('aria-expanded', 'true');

        /*---------------------------------------------------------------------
        FullScreen
        -----------------------------------------------------------------------*/
        jQuery(document).on('click', '.full-screen', function() {
            let elem = jQuery(this);
            elem.find('i').addClass('d-none');
            elem.find('i').addClass('d-none');
            if (!document.fullscreenElement &&
                !document.mozFullScreenElement && // Mozilla
                !document.webkitFullscreenElement && // Webkit-Browser
                !document.msFullscreenElement) { // MS IE ab version 11
                    elem.find('.min').removeClass('d-none');
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen();
                } else if (document.documentElement.mozRequestFullScreen) {
                    document.documentElement.mozRequestFullScreen();
                } else if (document.documentElement.webkitRequestFullscreen) {
                    document.documentElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
                } else if (document.documentElement.msRequestFullscreen) {
                    document.documentElement.msRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
                }
            } else {
                elem.find('.max').removeClass('d-none');
                if (document.cancelFullScreen) {
                    document.cancelFullScreen();
                } else if (document.mozCancelFullScreen) {
                    document.mozCancelFullScreen();
                } else if (document.webkitCancelFullScreen) {
                    document.webkitCancelFullScreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        });


        /*---------------------------------------------------------------------
        Page Loader
        -----------------------------------------------------------------------*/
        jQuery("#load").fadeOut();
        jQuery("#loading").delay().fadeOut("");


        /*---------------------------------------------------------------------
        Counter
        -----------------------------------------------------------------------*/
        if (window.counterUp !== undefined) {
            const counterUp = window.counterUp["default"]
            const $counters = $(".counter");
            $counters.each(function (ignore, counter) {
                var waypoint = new Waypoint( {
                    element: $(this),
                    handler: function() {
                        counterUp(counter, {
                            duration: 1000,
                            delay: 10
                        });
                        this.destroy();
                    },
                    offset: 'bottom-in-view',
                } );
            });
        }


        /*---------------------------------------------------------------------
        Progress Bar
        -----------------------------------------------------------------------*/
        jQuery('.iq-progress-bar > span').each(function() {
            let progressBar = jQuery(this);
            let width = jQuery(this).data('percent');
            progressBar.css({
                'transition': 'width 2s'
            });
            setTimeout(function() {
                    progressBar.css('width', width + '%');
            }, 100);
        });

        jQuery('.progress-bar-vertical > span').each(function () {
            let progressBar = jQuery(this);
            let height = jQuery(this).data('percent');
            progressBar.css({
                'transition': 'height 2s'
            });
            setTimeout(function () {
                    progressBar.css('height', height + '%');
            }, 100);
        });


        /*---------------------------------------------------------------------
        Page Menu
        -----------------------------------------------------------------------*/
        jQuery(document).on('click', '.wrapper-menu', function() {
            jQuery(this).toggleClass('open');
        });

        jQuery(document).on('click', ".wrapper-menu", function() {
            jQuery("body").toggleClass("sidebar-main");
        });


      /*---------------------------------------------------------------------
       Close  navbar Toggle
       -----------------------------------------------------------------------*/

        jQuery('.close-toggle').on('click', function () {
            jQuery('.h-collapse.navbar-collapse').collapse('hide');
        });

        /*---------------------------------------------------------------------
        user toggle
        -----------------------------------------------------------------------*/
        jQuery(document).on('click', '.user-toggle', function() {
            jQuery(this).parent().addClass('show-data');
        });

        jQuery(document).on('click', ".close-data", function() {
            jQuery('.user-toggle').parent().removeClass('show-data');
        });
        jQuery(document).on("click", function(event){
        var $trigger = jQuery(".user-toggle");
        if($trigger !== event.target && !$trigger.has(event.target).length){
            jQuery(".user-toggle").parent().removeClass('show-data');
        }
        });

        /*---------------------------------------------------------------------
        Data tables
        -----------------------------------------------------------------------*/
        if($.fn.DataTable){
            const table = $('.data-table').DataTable();
        }


        /*---------------------------------------------------------------------
        Form Validation
        -----------------------------------------------------------------------*/

        // Example starter JavaScript for disabling form submissions if there are invalid fields
        window.addEventListener('load', function() {
            // Fetch all the forms we want to apply custom Bootstrap validation styles to
            var forms = document.getElementsByClassName('needs-validation');
            // Loop over them and prevent submission
            var validation = Array.prototype.filter.call(forms, function(form) {
                form.addEventListener('submit', function(event) {
                    if (form.checkValidity() === false) {
                        event.preventDefault();
                        event.stopPropagation();
                    }
                    form.classList.add('was-validated');
                }, false);
            });
        }, false);

      /*---------------------------------------------------------------------
       Active Class for Pricing Table
       -----------------------------------------------------------------------*/
      jQuery("#my-table tr th").click(function () {
        jQuery('#my-table tr th').children().removeClass('active');
        jQuery(this).children().addClass('active');
        jQuery("#my-table td").each(function () {
          if (jQuery(this).hasClass('active')) {
            jQuery(this).removeClass('active')
          }
        });
        var col = jQuery(this).index();
        jQuery("#my-table tr td:nth-child(" + parseInt(col + 1) + ")").addClass('active');
      });


        /*---------------------------------------------------------------------
        Scrollbar
        -----------------------------------------------------------------------*/

        jQuery('.data-scrollbar').each(function () {
            var attr = $(this).attr('data-scroll');
            if (typeof attr !== typeof undefined && attr !== false){
            let Scrollbar = window.Scrollbar;
            var a = jQuery(this).data('scroll');
            Scrollbar.init(document.querySelector('div[data-scroll= "' + a + '"]'));
            }
        });


      /*---------------------------------------------------------------------
      image-upload
      -----------------------------------------------------------------------*/

      $('.form_gallery-upload').on('change', function() {
          var length = $(this).get(0).files.length;
          var galleryLabel  = $(this).attr('data-name');

          if( length > 1 ){
            $(galleryLabel).text(length + " files selected");
          } else {
            $(galleryLabel).text($(this)[0].files[0].name);
          }
        });

    /*---------------------------------------------------------------------
        video
      -----------------------------------------------------------------------*/
      $(document).ready(function(){
      $('.form_video-upload input').change(function () {
        $('.form_video-upload p').text(this.files.length + " file(s) selected");
      });
    });
    /*---------------------------------------------------------------------
        dark mode
      -----------------------------------------------------------------------*/
      const urlParams = new URLSearchParams(window.location.search);
      const mode = urlParams.get('dark');
      if (mode !== null) {
          $('body').removeClass('sidebar-dark', 'sidebar-light')
          switch (mode) {
              case "true":
                  $('body').addClass('dark')
              break;
              case "false":
                  $('body').removeClass('sidebar-dark', 'sidebar-light')
              break;
              default:
                  $('body').removeClass('sidebar-dark').removeClass('sidebar-light')
                  break;
          }
      }


        /*---------------------------------------------------------------------
        Button
        -----------------------------------------------------------------------*/

        jQuery('.qty-btn').on('click',function(){
          var id = jQuery(this).attr('id');

          var val = parseInt(jQuery('#quantity').val());

          if(id == 'btn-minus')
          {
            if(val != 0)
            {
              jQuery('#quantity').val(val-1);
            }
            else
            {
              jQuery('#quantity').val(0);
            }

          }
          else
          {
            jQuery('#quantity').val(val+1);
          }
        });

    });


    $(document).on('click', '[data-toggel-extra="side-nav"]', function () {
        const pannel = $(this).attr('data-expand-extra')
        $(pannel).addClass('active')
    })

    $(document).on('click', '[data-toggel-extra="side-nav-close"]', function () {
        const pannel = $(this).attr('data-expand-extra')
        $(pannel).removeClass('active')
    })

    $(document).on('click', '[data-toggel-extra="right-sidenav"]', function () {
        const target = $(this).data('target')
        $(target).addClass('active')
    })

    $(document).on('click', '[data-extra-dismiss="right-sidenav"]', function () {
        $(this).closest('.right-sidenav').removeClass('active')
    })

    $(document).on('click', '[data-toggle="end-call"]', function(){
        $(this).closest('.tab-pane').removeClass('active').removeClass('show')
        $($(this).attr('data-target')).tab('show')
        $('.chat-action').find('[data-toggle="tab"]').removeClass('active')
    })

    $(document).on('click', '[data-toggle-extra="tab"]', function () {
        const target = $(this).attr('data-target-extra')
        $('[data-toggle-extra="tab-content"]').removeClass('active')
        $(target).addClass('active')
        $(this).parent().find('.active').removeClass('active')
        $(this).addClass('active')
    })

    $('emoji-picker').on('emoji-click', function(e){
        $(e.target.dataset.targetInput).val($(e.target.dataset.targetInput).val()+e.detail.unicode)
    })

    $('.dropdown-menu').on('click', function(event){
        event.stopPropagation();
    });

    var board =  $('.draggable-item');

    var selector = [];
    if(board.length > 0 )
    {
        for(var i = 0 ; i < board.length ; i++) {
            selector.push(document.querySelector('#draggable-item-'+i));
            selector.push(document.querySelector('#list-draggable-item-'+i));
        }
    }
    dragula( selector ).on('drop', function(el) {
        $(el).addClass(' animate__animated animate__rubberBand')
        setTimeout(function () { 
            $(el).removeClass(' animate__animated animate__rubberBand')
        }, 1000)
    });

    // calender 1 js
    var calendar1;
    if (jQuery('#calendar1').length) {
        var calendarEl = document.getElementById('calendar1');

            calendar1 = new FullCalendar.Calendar(calendarEl, {
                selectable: true,
                plugins: ["timeGrid", "dayGrid", "list", "interaction"],
                timeZone: "UTC",
                defaultView: "dayGridMonth",
                contentHeight: "auto",
                eventLimit: true,
                dayMaxEvents: 4,
                header: {
                    left: "prev,next today",
                    center: "title",
                    right: "dayGridMonth,timeGridWeek,timeGridDay,listWeek"
                },
                dateClick: function (info) {
                    $('#schedule-start-date').val(info.dateStr)
                    $('#schedule-end-date').val(info.dateStr)
                    $('#date-event').modal('show')
                },
                events: [
                    {
                        title: 'Click for Google',
                        url: 'http://google.com/',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-20, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#4731b6'
                    },
                    {
                        title: 'All Day Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-18, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#465af7'
                    },
                    {
                        title: 'Long Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-16, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        end: moment(new Date(), 'YYYY-MM-DD').add(-13, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#7858d7'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-14, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#465af7'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-12, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#5baa73'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-10, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#01041b'
                    },
                    {
                        title: 'Birthday Party',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-8, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#4731b6'
                    },
                    {
                        title: 'Meeting',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-6, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#15ca92'
                    },
                    {
                        title: 'Birthday Party',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-5, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#f4a965'
                    },
                    {
                        title: 'Birthday Party',
                        start: moment(new Date(), 'YYYY-MM-DD').add(-2, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#ea643f'
                    },

                    {
                        title: 'Meeting',
                        start: moment(new Date(), 'YYYY-MM-DD').add(0, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#15ca92'
                    },
                    {
                        title: 'Click for Google',
                        url: 'http://google.com/',
                        start: moment(new Date(), 'YYYY-MM-DD').add(0, 'days').format('YYYY-MM-DD') + 'T06:30:00.000Z',
                        color: '#4731b6'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(0, 'days').format('YYYY-MM-DD') + 'T07:30:00.000Z',
                        color: '#5baa73'
                    },
                    {
                        title: 'Birthday Party',
                        start: moment(new Date(), 'YYYY-MM-DD').add(0, 'days').format('YYYY-MM-DD') + 'T08:30:00.000Z',
                        color: '#f4a965'
                    },
                    {
                        title: 'Doctor Meeting',
                        start: moment(new Date(), 'YYYY-MM-DD').add(0, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#f4a965'
                    },
                    {
                        title: 'All Day Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(1, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#465af7'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(8, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#465af7'
                    },
                    {
                        groupId: '999',
                        title: 'Repeating Event',
                        start: moment(new Date(), 'YYYY-MM-DD').add(10, 'days').format('YYYY-MM-DD') + 'T05:30:00.000Z',
                        color: '#5baa73'
                    }
                ]
            });
            calendar1.render();

        $(document).on("submit", "#submit-schedule", function (e) {
            e.preventDefault()
            const title = $(this).find('#schedule-title').val()
            const startDate = moment(new Date($(this).find('#schedule-start-date').val()), 'YYYY-MM-DD').format('YYYY-MM-DD') + 'T05:30:00.000Z'
            const endDate = moment(new Date($(this).find('#schedule-end-date').val()), 'YYYY-MM-DD').format('YYYY-MM-DD') + 'T05:30:00.000Z'
            const color = $(this).find('#schedule-color').val()
            const event = {
                title: title,
                start: startDate || '2020-12-22T02:30:00',
                end: endDate || '2020-12-12T14:30:00',
                color: color || '#7858d7'
            }
            $(this).closest('#date-event').modal('hide')
            calendar1.addEvent(event)
        })
    }

    const progressBar = document.getElementsByClassName('circle-progress')
    Array.from(progressBar, (elem) => {
        const minValue = elem.getAttribute('data-min-value')
        const maxValue = elem.getAttribute('data-max-value')
        const value = elem.getAttribute('data-value')
        const  type = elem.getAttribute('data-type')
        if (elem.getAttribute('id') !== '' && elem.getAttribute('id') !== null) {
            new CircleProgress('#'+elem.getAttribute('id'), {
            min: minValue,
        max: maxValue,
        value: value,
        textFormat: type,
        });
        }
    })
    /*---------------------------------------------------------------------
            Vanila Datepicker
    -----------------------------------------------------------------------*/
    const datepickers = document.querySelectorAll('.vanila-datepicker')
    Array.from(datepickers, (elem) => {
    new Datepicker(elem)
    })
    const daterangePickers = document.querySelectorAll('.vanila-daterangepicker')
    Array.from(daterangePickers, (elem) => {
    new DateRangePicker(elem)
    })

    /*---------------------------------------------------------------------
            Choies.js
    -----------------------------------------------------------------------*/
    const choies = document.querySelectorAll('.choicesjs')
    Array.from(choies,(elem) => {
        new Choices(elem, {
            removeItemButton: true,
        })
    })
})(jQuery);
