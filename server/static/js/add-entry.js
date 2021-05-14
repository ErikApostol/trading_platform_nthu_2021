$(document).ready(function(){
    $("#add").click(function(e){
        event.preventDefault()
        $('#items').append('<div class="form-group">'+
                           '    <select id="asset_ticker" name="asset_ticker" style="font-size:1.5em;">'+
                           '        <option value="SPY">SPY   &#x2614; <!--rain--></option>'+
                           '        <option value="0050">0050 &#x1F31E;<!--sun--></option>'+
                           '        <option value="BIV">BIV   &#x2601; <!--cloud--></option>'+
                           '        <option value="LQD">LQD   &#x2614; </option>'+
                           '        <option value="MUB">MUB   &#x1F31E;</option>'+
                           '        <option value="TLT">TLT   &#x2601; </option>'+
                           '        <option value="VB">VB     &#x2614; </option>'+
                           '        <option value="VNQ">VNQ   &#x1F31E;</option>'+
                           '        <option value="VOO">VOO   &#x2601; </option>'+
                           '        <option value="VEA">VEA   &#x2614; </option>'+
                           '        <option value="VWO">VWO   &#x1F31E;</option>'+
                           '        <option value="IAU">IAU   &#x2601; </option>'+
                           '        <option value="V">V       &#x1F31E;</option>'+
                           '        <option value="APPL">APPL &#x2601; </option>'+
                           '        <option value="AMZN">AMZN &#x2614; </option>'+
                           '        <option value="EMB">EMB   &#x1F31E;</option>'+
                           '        <option value="GLD">GLD   &#x2601; </option>'+
                           '    </select>'+
                           '    <input type="button" value="delete" id="delete" class="btn btn-danger"/>'+
                           '</div>'
        );
    });

    $('body').on('click', '#delete', function(e){
        $(this).parent('div').remove();
    });

});
