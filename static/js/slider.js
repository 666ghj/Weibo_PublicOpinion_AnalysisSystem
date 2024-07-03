/*
Template: Datum - Responsive Bootstrap 4 Admin Dashboard Template
Author: iqonic.design
Design and Developed by: http://iqonic.design/
NOTE: This file contains the styling for Slider in Template.
*/


jQuery(document).ready(function() {
  if(typeof $.fn.slick !== typeof undefined){
    /*---------------------------------------------------------------------
      slick
      -----------------------------------------------------------------------*/
      jQuery('.slick-slider').slick({
        centerMode: true,
        centerPadding: '60px',
        slidesToShow: 9,
        slidesToScroll: 1,
        focusOnSelect: true,
        responsive: [{
            breakpoint: 992,
            settings: {
                arrows: false,
                centerMode: true,
                centerPadding: '30',
                slidesToShow: 3
            }
        }, {
            breakpoint: 480,
            settings: {
                arrows: false,
                centerMode: true,
                centerPadding: '15',
                slidesToShow: 1
            }
        }],
        nextArrow: '<a href="#" class="ri-arrow-left-s-line left"></a>',
        prevArrow: '<a href="#" class="ri-arrow-right-s-line right"></a>',
    });

    jQuery('.top-rated-item').slick({
        slidesToShow: 4,
        speed: 300,
        slidesToScroll: 1,
        focusOnSelect: true,
         appendArrows: jQuery('#top-rated-item-slick-arrow'),
        responsive: [{
            breakpoint: 1200,
            settings: {
                slidesToShow: 3
            }
        },{
            breakpoint: 798,
            settings: {
                slidesToShow: 2
            }
        },{
            breakpoint: 480,
            settings: {
                arrows: false,
                autoplay:true,
                slidesToShow: 1
            }
        }],
    });

    jQuery('#newrealease-slider').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 300,
      centerMode: true,
      centerPadding: false,
      variableWidth: true ,
      infinite: true,
      focusOnSelect: true,
      autoplay: false,
      slidesToShow: 7,
      slidesToScroll: 1,


    });

   jQuery("#newrealease-slider .slick-active.slick-center").prev('.slick-active').addClass('temp');
    jQuery("#newrealease-slider .slick-active.temp").prev().addClass('temp-1');
    jQuery("#newrealease-slider .slick-active.temp-1").prev().addClass('temp-2');

     jQuery("#newrealease-slider .slick-active.slick-center").next('.slick-active').addClass('temp-next');
    jQuery("#newrealease-slider .slick-active.temp-next").next().addClass('temp-next-1');
    jQuery("#newrealease-slider .slick-active.temp-next-1").next().addClass('temp-next-2');

     jQuery("#newrealease-slider").on("afterChange", function (){
      var slick_index = jQuery(".slick-active.slick-center").data('slick-index');

      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-1)+'"]').addClass('temp');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-2)+'"]').addClass('temp-1');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-3)+'"]').addClass('temp-2');

      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+1))+'"]').addClass('temp-next');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+2))+'"]').addClass('temp-next-1');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+3))+'"]').addClass('temp-next-2');


    });

    jQuery("#newrealease-slider").on("beforeChange", function (){
      var slick_index = jQuery(".slick-active.slick-center").data('slick-index');

      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-1)+'"]').removeClass('temp');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-2)+'"]').removeClass('temp-1');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(slick_index-3)+'"]').removeClass('temp-2');

      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+1))+'"]').removeClass('temp-next');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+2))+'"]').removeClass('temp-next-1');
      jQuery('#newrealease-slider .slick-active[data-slick-index="'+(parseInt(slick_index+3))+'"]').removeClass('temp-next-2');

    });

    jQuery('#favorites-slider').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 300,
      centerMode: false,
      autoplay: true,
      slidesToShow: 3,
      slidesToScroll: 1,
      responsive: [
        {
          breakpoint: 1024,
          settings: {
            slidesToShow: 3,
            slidesToScroll: 1,
            infinite: true,
          }
        },
        {
          breakpoint: 992,
          settings: {
            slidesToShow: 2,
            slidesToScroll: 1
          }
        },
        {
          breakpoint: 767,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1
          }
        }
      ]
    });

    jQuery('#similar-slider').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 300,
      centerMode: false,
      autoplay: true,
      slidesToShow: 4,
      slidesToScroll: 1,
      responsive: [
        {
          breakpoint: 1024,
          settings: {
            slidesToShow: 3,
            slidesToScroll: 1,
            infinite: true,
          }
        },
        {
          breakpoint: 768,
          settings: {
            slidesToShow: 2,
            slidesToScroll: 1
          }
        },
        {
          breakpoint: 576,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1
          }
        }
      ]
    });

    jQuery('#single-similar-slider').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 300,
      centerMode: false,
      autoplay: true,
      slidesToShow: 3,
      slidesToScroll: 1,
      responsive: [
        {
          breakpoint: 1024,
          settings: {
            slidesToShow: 3,
            slidesToScroll: 1,
            infinite: true,
          }
        },
        {
          breakpoint: 768,
          settings: {
            slidesToShow: 2,
            slidesToScroll: 1
          }
        },
        {
          breakpoint: 576,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1
          }
        }
      ]
    });

    jQuery('#trendy-slider').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 300,
      centerMode: false,
      autoplay: true,
      slidesToShow: 4,
      slidesToScroll: 1,
      responsive: [
        {
          breakpoint: 1024,
          settings: {
            slidesToShow: 3,
            slidesToScroll: 1,
            infinite: true,
          }
        },
        {
          breakpoint: 768,
          settings: {
            slidesToShow: 2,
            slidesToScroll: 1
          }
        },
        {
          breakpoint: 576,
          settings: {
            slidesToShow: 1,
            slidesToScroll: 1
          }
        }
      ]
    });

    jQuery('#description-slider').slick({
      slidesToShow: 1,
      slidesToScroll: 1,
      arrows: false,
      fade: true,
      asNavFor: '#description-slider-nav'
    });

    jQuery('#description-slider-nav').slick({
      slidesToShow: 3,
      slidesToScroll: 1,
      asNavFor: '#description-slider',
      dots: false,
      arrows: false,
      infinite: true,
      vertical: true,
      centerMode: false,
      focusOnSelect: true
    });

    jQuery('.realeases-banner').slick({
      slidesToShow: 5,
      speed: 300,
      arrows:false,
      slidesToScroll: 1,
      vertical: true,
      verticalSwiping: true,
      focusOnSelect: true,
      responsive: [{
          breakpoint: 992,
          settings: {
              arrows: false,
              slidesToShow: 3
          }
      }, {
          breakpoint: 480,
          settings: {
              arrows: false,
              verticalSwiping:false,
              slidesToShow:4
          }
      }],
  });

  jQuery('.feature-album').slick({
      slidesToShow: 6,
      speed: 300,
      slidesToScroll: 1,
      focusOnSelect: true,
       appendArrows: jQuery('#feature-album-slick-arrow'),
      responsive: [{
          breakpoint: 1200,
          settings: {
              slidesToShow: 4
          }
      },{
          breakpoint: 992,
          settings: {
              slidesToShow: 3
          }
      }, {
          breakpoint: 480,
          settings: {
              arrows: false,
              autoplay:true,
              slidesToShow: 1
          }
      }],
  });

 jQuery('.feature-album-artist').slick({
      slidesToShow: 6,
      speed: 300,
      slidesToScroll: 1,
      appendArrows: jQuery('#feature-album-artist-slick-arrow'),
      focusOnSelect: true,
      responsive: [{
          breakpoint: 1200,
          settings: {
              slidesToShow: 4
          }
      },{
          breakpoint: 992,
          settings: {
              arrows:true,
              slidesToShow: 3
          }
      }, {
          breakpoint: 480,
          settings: {
              arrows:false,
              autoplay:true,
              slidesToShow: 1
          }
      }],
  });

  jQuery('.hot-songs').slick({
      slidesToShow: 2,
      speed: 300,
      appendArrows: jQuery('#hot-song-slick-arrow'),
      slidesToScroll: 1,
      rows:3,
      focusOnSelect: true,
      responsive: [{
          breakpoint: 992,
          settings: {
              arrows: true,
              slidesToShow: 2
          }
      }, {
          breakpoint: 480,
          settings: {
              arrows: false,
              autoplay:true,
              slidesToShow: 1
          }
      }],
  });

    /*---slider salon----*/
      jQuery('.salone-styles').slick({
        dots: false,
        arrows: true,
        infinite: true,
        speed: 200,
        autoplay: false,
        slidesToShow: 2,
        slidesToScroll: 1,
        appendArrows: jQuery('#trending-order-slick-arrow'),
        responsive: [
          {
            breakpoint: 1200,
            settings: {
              slidesToShow: 1
            }
          },
          {
            breakpoint: 1024,
            settings: {
              slidesToShow: 1
            }
          },
          {
            breakpoint: 600,
            settings: {
              slidesToShow: 1
            }
          }
        ]
      });

   jQuery('.hot-video').slick({
      slidesToShow: 2,
      speed: 300,
      appendArrows: jQuery('#hot-video-slick-arrow'),
      slidesToScroll: 1,
      focusOnSelect: true,
      responsive: [{
          breakpoint: 992,
          settings: {
              arrows: true,
              slidesToShow: 2
          }
      }, {
          breakpoint: 480,
          settings: {
              arrows: false,
              autoplay:true,
              slidesToShow: 1
          }
      }],
  });

/*---------------------------------------------------------------------
active music
-----------------------------------------------------------------------*/
  jQuery( 'ul.iq-song-slide li').on('click', function(){
      jQuery('ul.iq-song-slide li').removeClass('active');
      jQuery(this).addClass('active');
  });


/*---------------------------------------------------------------------
social media post
-----------------------------------------------------------------------*/
    jQuery('.post-social').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 200,
      autoplay: true,
      slidesToShow: 1,
      slidesToScroll: 1,
    });

      jQuery('.trending-order').slick({
        dots: false,
        arrows: true,
        infinite: true,
        speed: 200,
        autoplay: false,
        slidesToShow: 2,
        slidesToScroll: 1,
        appendArrows: jQuery('#trending-order-slick-arrow'),
        responsive: [
          {
            breakpoint: 1300,
            settings: {
              slidesToShow: 2
            }
          },
          {
            breakpoint: 1024,
            settings: {
              slidesToShow: 2
            }
          },
          {
            breakpoint: 600,
            settings: {
              slidesToShow: 1
            }
          }
        ]
      });

      jQuery('.resto-blog').slick({
        dots: false,
        arrows: false,
        infinite: true,
        speed: 200,
        autoplay: false,
        slidesToShow: 2,
        slidesToScroll: 1,
        responsive: [
          {
            breakpoint: 600,
            settings: {
              slidesToShow: 1
            }
          }
        ]
      });

    jQuery('.image-slide-1').slick({
      dots: false,
      arrows: false,
      infinite: true,
      speed: 200,
      autoplay: true,
      slidesToShow: 1,
      slidesToScroll: 1,
    });

    jQuery('.stylist-salon').slick({
      slidesToShow: 4,
      speed: 300,
      slidesToScroll: 1,
      focusOnSelect: true,
      autoplay: true,
      arrows: false,
      responsive: [{
        breakpoint: 992,
        settings: {
          arrows: false,
          slidesToShow: 2
        }
      }, {
        breakpoint: 480,
        settings: {
          arrows: false,
          autoplay: true,
          slidesToShow: 1
        }
      }],
    });

    jQuery('.stylist-salon1').slick({
      slidesToShow: 4,
      speed: 300,
      slidesToScroll: 1,
      focusOnSelect: true,
      autoplay: true,
      responsive: [{
        breakpoint: 992,
        settings: {
          arrows: true,
          slidesToShow: 2
        }
      }, {
        breakpoint: 480,
        settings: {
          arrows: false,
          autoplay: true,
          slidesToShow: 1
        }
      }],
    });


  }
});
