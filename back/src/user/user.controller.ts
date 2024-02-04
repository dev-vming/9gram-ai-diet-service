
import { response, Request } from 'express';
import { UserService } from './user.service';
import { Body, Controller, Delete, Get, Param, Post, Put, Req, Res, UseFilters, UseGuards } from '@nestjs/common';
import { ApiBody, ApiOperation, ApiResponse, ApiResponseProperty, ApiTags } from '@nestjs/swagger';
import { isLoggedInGuard } from 'src/auth/utils/guards/isLoggedin.guard';
import { UpdateUserDto } from './dto/UpdateUser.dto';
import { SaveHealthInfoDto } from './dto/SaveHealthInfo.dto';
import { handleUpdateUserInfosDto } from './dto/handleUpdateUserInfos.dto';
import { ErrorToObjectFilter } from 'src/auth/utils/exception-filters/redirectFilter';

@ApiTags('user')
@Controller('user')
export class UserController {
    constructor(private userService: UserService){}


    @ApiOperation({summary: '유저정보와 유저건강정보 가져오기'})
    @UseFilters(ErrorToObjectFilter)
    @UseGuards(isLoggedInGuard)
    @Get()
    public async handleGetUserInfos(@Req() request: any, @Res() response: any){
        const result =  await this.userService.getUserInfos(request.user.userId);
        response.status(200).json(result);
    }

    @ApiOperation( {summary: '유저정보와 유저건강정보 수정하기'} )
    @ApiBody({ type: handleUpdateUserInfosDto })
    @UseFilters(ErrorToObjectFilter)
    @UseGuards(isLoggedInGuard)
    @Put()
    public async handleUpdateUserInfos(
        @Req() request: any, 
        @Body() updateUserInfosDto: UpdateUserDto | SaveHealthInfoDto, 
        @Res() response: any){
                console.log('사용자 정보 업데이트 시작...')
                await this.userService.updateUserInfos(request.user.userId, updateUserInfosDto);
                const result = await this.userService.getUserInfos(request.user.userId);
                response.status(200).send(result);
    }


    /* 유저네임 중복 확인하기 */
    @ApiOperation({ summary: '유저네임 중복 확인하기' })
    @Get('/username/:username')
    async handleCheckUsername(@Param('username') username:string, @Res() response: any): Promise<void> {
        try{
            const result = await this.userService.checkUsername(username);
            response.status(200).send({isAvailable: !result});
        }catch(err){ throw err; }
    }
}
