require "socket"

while true do
    dataFile = io.open("extracted_data.txt", "w+");
    
    io.output(dataFile);

    -- Get the data
    local marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86);
    local marioY = memory.readbyte(0x03B8) + 16;

    io.write("MarioPosition:" .. marioX .. "," .. marioY .. "\n");
    
    local screenX = memory.readbyte(0x03AD);
    local screenY = memory.readbyte(0x03B8);

    for slot=0,4 do
        local enemy = memory.readbyte(0xF+slot);
        if enemy ~= 0 then
            local ex = memory.readbyte(0x6E + slot) * 0x100 + memory.readbyte(0x87 + slot);
            local ey = memory.readbyte(0xCF + slot) + 24;
            io.write("Enemy" .. slot .. ":" .. ex .. "," .. ey .. "\n");
        end
    end

    io.write("Timestamp:" .. socket.gettime() * 1000); -- Timestamp of the data
    
    io.close(dataFile);

    -- client.screenshot("screenshot.png");

    local cmdFile = io.open("commands.txt", "r");
    io.input(cmdFile);

    joypad.set( 
        {
            Right = io.read() == "true" and true or false,
            Left = io.read() == "true" and true or false,
            Up = io.read() == "true" and true or false,
            Down = io.read() == "true" and true or false,
            A = io.read() == "true" and true or false,
            B = io.read() == "true" and true or false
        }, 1
    );

    emu.frameadvance();
end;