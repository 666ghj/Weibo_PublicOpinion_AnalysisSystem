
$(function () {

    /*---------------------------------------------------------------------
        Basic Tree
    -----------------------------------------------------------------------*/
    $('.basic-tree li:has(ul)').addClass('t-parent').find(' > span').attr('title', 'Collapse this branch');
    $('.basic-tree li.t-parent > span').on('click', function (e) {
        var children = $(this).parent('li.t-parent').find(' > ul > li');
        if (children.is(":visible")) {
            children.hide('fast');
            $(this).attr('title', 'Expand this branch').find(' > i').addClass('fa-plus').removeClass('fa-minus');
        } else {
            children.show('fast');
            $(this).attr('title', 'Collapse this branch').find(' > i').addClass('fa-minus').removeClass('fa-plus');
        }
        e.stopPropagation();
    });

    /*---------------------------------------------------------------------
        Checkbox Tree
    -----------------------------------------------------------------------*/
    jQuery('#flex-tree-check').flexTree({
        type: 'checkbox',
        name: 'foo[]',
        collapsed: false,
        items: [
            {
                label: 'Item 1',
                childrens: [
                    {
                        label: 'Item 1.1',
                        value: 'item_1_1',
                        checked: true
                    },

                    {
                        label: 'Item 1.2',
                        value: 'item_1_2',
                        childrens: [
                            {
                                label: 'Item 1.2.1',
                                value: 'item_1_2_1',
                                childrens: [
                                    {
                                        label: 'Item 1.2.2.1',
                                        value: 'item_1_2_2_1'
                                    },

                                    {
                                        label: 'Item 1.2.2.2',
                                        value: 'item_1_2_2_2',
                                        id: 'foo'
                                    }]
                            },
                            {
                                label: 'Item 1.2.2',
                                value: 'item_1_2_2'
                            }]
                    },
                    {
                        label: 'Item 1.3',
                        value: 'item_1_3',
                        checked: true
                    }]
            },
            {
                label: 'Item 2',
                childrens: [
                    {
                        label: 'Item 2.1',
                        value: 'item_2_1',
                        checked: true
                    }]
            }]
        });
    });

    


    